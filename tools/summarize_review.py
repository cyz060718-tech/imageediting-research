"""Aggregate manually reviewed rows with explicit denominators and missing counts."""
import argparse
import csv
from collections import defaultdict,Counter
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('review_csv');a=p.parse_args()
    path=Path(a.review_csv)
    with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    grouped=defaultdict(list)
    for row in rows:grouped[row['method']].append(row)
    output=[]
    for method,group in grouped.items():
        out={'method':method,'runs':len(group),'unique_images':len({r['image'] for r in group})}
        for key in ['semantic_success','count_preserved','identity_preserved','boundary_natural']:
            values=[r[key].strip() for r in group]
            invalid=[v for v in values if v not in {'','0','1','NA'}]
            if invalid:raise ValueError(f'{method} {key}: 仅接受空白/0/1/NA，发现 {invalid}')
            rated=[int(v) for v in values if v in {'0','1'}]
            out[key+'_n']=len(rated);out[key+'_rate']=sum(rated)/len(rated) if rated else ''
            out[key+'_missing']=values.count('');out[key+'_na']=values.count('NA')
        out['failure_counts']=str(dict(Counter(r['failure_type'] for r in group if r['failure_type'])))
        output.append(out)
    target=path.with_name('human_review_summary.csv')
    fields=list(dict.fromkeys(k for row in output for k in row))
    with target.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(output)
    print(target)

if __name__=='__main__':main()
