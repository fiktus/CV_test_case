import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).parent
fields = ["Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
          "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
          "Issue_organization_ru"]
name_fields = ['Last_name_ru', 'First_name_ru', 'Middle_name_ru']
two_fields = ['Birth_place_ru', 'Issue_organization_ru']
mrz_fields = ['Birth_date', 'Issue_date', 'Issue_organisation_code', 'Licence_number']
min_conf = {'web': 0.95, 'mobile': 0.95, 'mobile_v2': 0.9995}
protected = {'Р-НЕ', 'Р-НА', 'РЕСПУБЛИКЕ', 'ГОР.'}
on_date = pd.Timestamp('2026-10-01')
translit = {'А':'A','Б':'B','В':'V','Г':'G','Д':'D','Е':'E','Ё':'E','Ж':'J','З':'Z','И':'I',
    'Й':'Q','К':'K','Л':'L','М':'M','Н':'N','О':'O','П':'P','Р':'R','С':'S','Т':'T','У':'U','Ф':'F',
    'Х':'H','Ц':'C','Ч':'3','Ш':'4','Щ':'W','Ъ':'X','Ы':'Y','Ь':'9','Э':'6','Ю':'7','Я':'8',' ':'<','-':'<'}
inv_translit = {lat: cyr for cyr, lat in translit.items() if cyr not in 'Ё -'}

def norm(s):
    return " ".join(s.split()).upper() if isinstance(s, str) else ""

def load(path):
    return pd.json_normalize(pd.read_json(path, lines=True).to_dict('records')).set_index('doc_id')

def field_quality(box, lines):
    if not isinstance(lines, list):
        return np.nan, np.nan
    line_h = np.mean([l['box'][3] for l in lines])
    confs = []
    for l in lines:
        confs += l['char_conf']
    return min(confs), round(box[3] / line_h) - len(lines)

def text_conf(lines):
    if not isinstance(lines, list):
        return '', []
    conf = []
    for l in lines:
        conf += l['char_conf'] + [1.0]
    return ' '.join(l['text'] for l in lines), conf

def line_text(lines, i):
    if not isinstance(lines, list) or len(lines) <= i:
        return None
    return lines[i]['text']

def add_quality(df):
    for f in fields:
        df[f'viz.{f}.chars'] = df[f'viz.{f}.lines'].map(text_conf)
        df[f'viz.{f}.text'] = df[f'viz.{f}.chars'].map(lambda chars: norm(chars[0]))
    for f in two_fields:
        quality = [field_quality(box, lines) for box, lines in zip(df[f'viz.{f}.box'], df[f'viz.{f}.lines'])]
        df[f'viz.{f}.cmin'] = [q[0] for q in quality]
        df[f'viz.{f}.lost'] = [q[1] for q in quality]
    df['mrz.chars'] = df['mrz.lines'].map(text_conf)
    df['mrz.line0'] = df['mrz.lines'].map(lambda lines: line_text(lines, 0))
    df['mrz.line1'] = df['mrz.lines'].map(lambda lines: line_text(lines, 1))
    return df

def check(s, digit):
    s = s.replace('<', '0')
    if not s.isdigit():
        return False
    weights = [7, 3, 1]
    total = 0
    for i, c in enumerate(s):
        total += int(c) * weights[i % 3]
    return str(total % 10) == digit

def mrz_date(date, max_year):
    if not date.isdigit():
        return None
    year = 2000 + int(date[:2])
    return f'{date[4:]}.{date[2:4]}.{year if year <= max_year else year - 100}'

def parse_line2(t):
    if not isinstance(t, str) or len(t) != 44:
        return {}
    number_ok = check(t[0:9], t[9])
    birth_ok = check(t[13:19], t[19])
    extra_ok = check(t[28:42], t[42])
    return {
        'Sex_ru': {'M': 'МУЖ.', 'F': 'ЖЕН.'}.get(t[20]),
        'Birth_date': mrz_date(t[13:19], max_year=2012) if birth_ok else None,
        'Issue_date': mrz_date(t[29:35], max_year=2026) if extra_ok else None,
        'Issue_organisation_code': f'{t[35:38]}-{t[38:41]}' if extra_ok else None,
        'Licence_number': f'{t[0:2]} {t[2]}{t[28]} {t[3:9]}' if number_ok and extra_ok else None,
    }

def to_latin(s):
    return ''.join(translit.get(ch, ch) for ch in s)

def pick_char(a, b, c, cc, low):
    if a == b or (c >= low > cc):
        return a
    if c < low <= cc and a.isalpha() and b.isalpha():
        return b
    return None

def merge(word, conf, cand, cand_conf, low=0.95):
    if len(word) != len(cand):
        return None
    out = [pick_char(a, b, c, cc, low) for a, b, c, cc in zip(word, cand, conf, cand_conf)]
    return None if None in out else ''.join(out)

def mrz_names(line, conf):
    if not isinstance(line, str) or len(line) != 44 or not line.endswith('<'):
        return None
    parts = line[5:].rstrip('<').split('<<', 1)
    last = parts[0]
    rest = parts[1] if len(parts) > 1 else ''
    names = ([last] + rest.split('<') + [''])[:3]
    if not all(c in inv_translit for c in ''.join(names)):
        return None
    starts = [5, 7 + len(names[0]), 8 + len(names[0]) + len(names[1])]
    return [(''.join(inv_translit[c] for c in s), conf[i:i + len(s)]) for s, i in zip(names, starts)]

def yymmdd(date):
    return date[8:10] + date[3:5] + date[:2]

def viz_line2(t, v):
    if not isinstance(t, str):
        return None
    n, c = v['Licence_number'].replace(' ', ''), v['Issue_organisation_code'].replace('-', '')
    return n[:3] + n[4:] + t[9:13] + yymmdd(v['Birth_date']) + t[19:28] + n[3:4] + yymmdd(v['Issue_date']) + c + t[41:]

def merge_names(df, doc):
    mrz = mrz_names(df.loc[doc, 'mrz.line0'], df.loc[doc, 'mrz.chars'][1])
    if mrz is None:
        return None
    merged = []
    for i, f in enumerate(name_fields):
        text, conf = df.loc[doc, f'viz.{f}.chars']
        cand, cand_conf = mrz[i]
        merged.append(merge(text.replace('Ё', 'Е'), conf, cand, cand_conf))
    return None if None in merged else merged

def fix_names(df, values):
    for f in name_fields:
        values[f] = values[f].str.replace('Ё', 'Е')
    names = (values['Last_name_ru'] + '<<' + values['First_name_ru'] + '<' + values['Middle_name_ru']).map(to_latin)
    names_ok = df['mrz.line0'].str[5:] == (names + '<' * 39).str[:39]
    for doc in df.index[~names_ok]:
        merged = merge_names(df, doc)
        if merged is not None:
            values.loc[doc, name_fields] = merged
            names_ok.loc[doc] = True
    return names_ok

def fix_numbers(df, values, mrz2):
    viz_lines = [viz_line2(df.loc[doc, 'mrz.line1'], values.loc[doc]) for doc in df.index]
    mrz_viz = pd.DataFrame([parse_line2(t) for t in viz_lines], index=df.index)
    numbers_ok = pd.DataFrame(index=df.index)
    for f in mrz_fields:
        numbers_ok[f] = mrz2[f].notna() | (mrz_viz[f] == values[f])
        values[f] = mrz2[f].fillna(values[f])
    return numbers_ok

def mrz_fix(df):
    values = pd.DataFrame({f: df[f'viz.{f}.text'] for f in fields})
    mrz2 = pd.DataFrame(df['mrz.line1'].apply(parse_line2).tolist(), index=df.index)
    names_ok = fix_names(df, values)
    covered = pd.DataFrame({f: names_ok for f in name_fields})
    covered['Sex_ru'] = values['Sex_ru'] == mrz2['Sex_ru']
    numbers_ok = fix_numbers(df, values, mrz2)
    return values, pd.concat([covered, numbers_ok], axis=1)

def fix_org(chars, known, known_by_len):
    text, conf = chars
    words, pos = [], 0
    for w in text.split(' '):
        c = conf[pos:pos + len(w)]
        if w in known or w in protected:
            cands = [w]
        else:
            cands = [v for v in known_by_len.get(len(w), []) if merge(w, c, v, [1.0] * len(v))]
        words.append(cands[0] if len(cands) == 1 else w)
        pos += len(w) + 1
    fixed = ' '.join(words)
    return fixed if fixed != text and all(w in known for w in words) else None

def is_valid(values):
    birth = pd.to_datetime(values['Birth_date'], format='%d.%m.%Y', errors='coerce')
    issue = pd.to_datetime(values['Issue_date'], format='%d.%m.%Y', errors='coerce')
    valid = (issue <= on_date) & (issue >= birth + pd.DateOffset(years=14))
    for age in [20, 45]:
        birthday = birth + pd.DateOffset(years=age)
        valid &= (issue >= birthday) | (birthday > on_date)
    return valid

def accept_rule(df, covered, values):
    ok = covered.all(axis=1) & is_valid(values)
    for f in two_fields:
        ok &= (df[f'viz.{f}.cmin'] >= df['source'].map(min_conf)) & (df[f'viz.{f}.lost'] == 0)
    return ok

def main(src, dst):
    subdivisions = pd.read_csv(root / 'reference/subdivisions.csv')
    known = set(' '.join(subdivisions['name']).upper().split())
    known_by_len = {}
    for v in known:
        if len(v) not in known_by_len:
            known_by_len[len(v)] = []
        known_by_len[len(v)].append(v)

    df = add_quality(load(src))
    values, covered = mrz_fix(df)
    fixed = df['viz.Issue_organization_ru.chars'].map(lambda chars: fix_org(chars, known, known_by_len))
    values['Issue_organization_ru'] = fixed.fillna(values['Issue_organization_ru'])
    df.loc[fixed.notna(), 'viz.Issue_organization_ru.cmin'] = 1.0
    accept = accept_rule(df, covered, values)

    with open(dst, 'w', encoding='utf-8') as out:
        for doc in df.index:
            answer = {'doc_id': doc, 'decision': 'accept' if accept[doc] else 'reject'}
            if accept[doc]:
                answer['fields'] = values.loc[doc, fields].to_dict()
            out.write(json.dumps(answer, ensure_ascii=False) + '\n')
    print('done')



main(sys.argv[1], sys.argv[2])
