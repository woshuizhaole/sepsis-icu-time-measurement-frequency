#!/usr/bin/env python3
"""Stage 1 schema and raw-file audit for CinC2019/CinC2012."""
from __future__ import annotations
import csv, json, os, re, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path('/data/workplace/wyx/measurement_intensity_ssl_20261015')
RAW = PROJECT / 'data' / 'raw'
OUT_STATS = PROJECT / 'results' / 'stats'
OUT_TABLES = PROJECT / 'results' / 'tables'
LOG_DIR = PROJECT / 'logs' / 'qc'
for p in (OUT_STATS, OUT_TABLES, LOG_DIR): p.mkdir(parents=True, exist_ok=True)

MISSING = {'', 'nan', 'NaN', 'NAN', 'null', 'NULL'}

def p19_one(path: Path):
    rec = {'path': str(path), 'split': path.parent.name, 'patient': path.stem,
           'rows': 0, 'positive_rows': 0, 'malformed_rows': 0, 'duplicate_iculos': 0,
           'min_iculos': None, 'max_iculos': None, 'headers': None,
           'missing': Counter(), 'nonmissing': Counter()}
    try:
        with path.open('r', newline='') as fh:
            header_line = fh.readline().rstrip('\n\r')
            header = header_line.split('|')
            rec['headers'] = header
            idx = {x: i for i, x in enumerate(header)}
            seen_iculos = set()
            i_iculos = idx.get('ICULOS')
            i_label = idx.get('SepsisLabel')
            for line in fh:
                row = line.rstrip('\n\r').split('|')
                if len(row) != len(header):
                    rec['malformed_rows'] += 1
                    continue
                rec['rows'] += 1
                for i, val in enumerate(row):
                    name = header[i]
                    if val in MISSING:
                        rec['missing'][name] += 1
                    else:
                        rec['nonmissing'][name] += 1
                if i_label is not None and row[i_label].strip() == '1':
                    rec['positive_rows'] += 1
                if i_iculos is not None:
                    try:
                        iculos = float(row[i_iculos])
                        if rec['min_iculos'] is None or iculos < rec['min_iculos']: rec['min_iculos'] = iculos
                        if rec['max_iculos'] is None or iculos > rec['max_iculos']: rec['max_iculos'] = iculos
                        if iculos in seen_iculos: rec['duplicate_iculos'] += 1
                        seen_iculos.add(iculos)
                    except ValueError:
                        pass
    except Exception as e:
        rec['error'] = repr(e)
    rec['missing'] = dict(rec['missing']); rec['nonmissing'] = dict(rec['nonmissing'])
    return rec

def p12_one(path: Path):
    rec = {'path': str(path), 'set': path.parent.name, 'patient': path.stem,
           'rows': 0, 'malformed_rows': 0, 'duplicate_events': 0,
           'min_time_min': None, 'max_time_min': None, 'parameters': Counter()}
    seen = set()
    try:
        with path.open('r', errors='replace') as fh:
            for line in fh:
                line = line.rstrip('\n\r')
                if not line: continue
                if line.startswith('Time,Parameter,Value'): continue
                parts = line.split(',', 2)
                if len(parts) != 3:
                    rec['malformed_rows'] += 1; continue
                t, param, value = parts
                m = re.match(r'^(\d{1,3}):(\d{2})$', t.strip())
                if not m:
                    rec['malformed_rows'] += 1; continue
                minutes = int(m.group(1))*60 + int(m.group(2))
                rec['rows'] += 1
                rec['parameters'][param] += 1
                if rec['min_time_min'] is None or minutes < rec['min_time_min']: rec['min_time_min'] = minutes
                if rec['max_time_min'] is None or minutes > rec['max_time_min']: rec['max_time_min'] = minutes
                key = (t, param)
                if key in seen: rec['duplicate_events'] += 1
                seen.add(key)
    except Exception as e:
        rec['error'] = repr(e)
    rec['parameters'] = dict(rec['parameters'])
    return rec

def parallel_map(func, paths, workers=32):
    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(func, p) for p in paths]
        for fut in as_completed(futures): out.append(fut.result())
    return sorted(out, key=lambda x: x['path'])

def write_csv(path, rows, fields):
    with path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows: w.writerow({k: r.get(k, '') for k in fields})

start = time.time()
# CinC2019
p19_paths = sorted((RAW/'cinc2019'/'training'/'training_setA').glob('*.psv')) + sorted((RAW/'cinc2019'/'training'/'training_setB').glob('*.psv'))
p19 = parallel_map(p19_one, p19_paths)
p19_headers = Counter(tuple(r.get('headers') or []) for r in p19)
p19_header = list(p19_headers.most_common(1)[0][0]) if p19_headers else []
p19_rows = sum(r['rows'] for r in p19)
p19_pos = sum(r['positive_rows'] for r in p19)
p19_malformed = sum(r['malformed_rows'] for r in p19)
p19_errors = [r for r in p19 if r.get('error')]
p19_per = []
for r in p19:
    p19_per.append({k:r.get(k,'') for k in ['patient','split','rows','positive_rows','malformed_rows','duplicate_iculos','min_iculos','max_iculos','path']})
write_csv(OUT_TABLES/'stage1_cinc2019_per_patient.csv', p19_per, ['patient','split','rows','positive_rows','malformed_rows','duplicate_iculos','min_iculos','max_iculos','path'])
p19_var = []
allvars = sorted(set(p19_header))
for var in allvars:
    p19_var.append({'variable':var, 'total_rows':p19_rows,
                    'missing_rows':sum(r.get('missing',{}).get(var,0) for r in p19),
                    'nonmissing_rows':sum(r.get('nonmissing',{}).get(var,0) for r in p19)})
write_csv(OUT_TABLES/'stage1_cinc2019_variable_stats.csv', p19_var, ['variable','total_rows','missing_rows','nonmissing_rows'])

# CinC2012 phase files
p12_paths = sorted((RAW/'cinc2012').glob('set-*/*.txt'))
p12 = parallel_map(p12_one, p12_paths)
p12_rows = sum(r['rows'] for r in p12)
p12_malformed = sum(r['malformed_rows'] for r in p12)
p12_dups = sum(r['duplicate_events'] for r in p12)
p12_per = [{k:r.get(k,'') for k in ['patient','set','rows','malformed_rows','duplicate_events','min_time_min','max_time_min','path']} for r in p12]
write_csv(OUT_TABLES/'stage1_cinc2012_per_patient.csv', p12_per, ['patient','set','rows','malformed_rows','duplicate_events','min_time_min','max_time_min','path'])
p12_param = Counter()
for r in p12: p12_param.update(r.get('parameters',{}))
write_csv(OUT_TABLES/'stage1_cinc2012_parameter_stats.csv', [{'parameter':k,'event_count':v} for k,v in sorted(p12_param.items())], ['parameter','event_count'])

# Outcome files and headers
outcome_info=[]
for f in sorted((RAW/'cinc2012').glob('Outcomes-*.txt')):
    with f.open(errors='replace') as fh:
        lines=fh.read().splitlines()
    outcome_info.append({'file':str(f),'lines':len(lines),'header':lines[0] if lines else ''})

summary = {
    'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'cinc2019': {
        'patient_files': len(p19), 'expected_training_patients': 40336,
        'split_counts': dict(Counter(r['split'] for r in p19)),
        'headers_unique': len(p19_headers), 'columns': p19_header,
        'rows': p19_rows, 'positive_label_rows': p19_pos,
        'positive_label_rate_rows': p19_pos / p19_rows if p19_rows else None,
        'malformed_rows': p19_malformed, 'error_files': len(p19_errors),
        'expected_fields_present': all(x in p19_header for x in ['SepsisLabel','ICULOS','HospAdmTime']),
        'duplicate_iculos_total': sum(r['duplicate_iculos'] for r in p19),
        'qc': 'PASS' if len(p19)==40336 and len(p19_headers)==1 and p19_malformed==0 and not p19_errors else 'PARTIAL'
    },
    'cinc2012': {
        'patient_files': len(p12), 'expected_primary_patient_files': 8000, 'phase1_duplicate_files': len(list((RAW/'cinc2012').glob('phase1/set-*/*.txt'))),
        'set_counts': dict(Counter(r['set'] for r in p12)),
        'rows': p12_rows, 'parameters': len(p12_param),
        'malformed_rows': p12_malformed, 'duplicate_events': p12_dups,
        'outcome_files': outcome_info,
        'qc': 'PASS' if len(p12)==8000 and p12_malformed==0 else 'PARTIAL'
    },
    'mimic_iv_demo': {'download_attempted': True, 'files': 0, 'status': 'PARTIAL_401'},
    'elapsed_seconds': round(time.time()-start,2)
}
(OUT_STATS/'stage1_schema_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))
