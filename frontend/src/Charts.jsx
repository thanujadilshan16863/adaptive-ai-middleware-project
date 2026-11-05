import React from 'react';
const colors={Smooth:'#247c68',Erratic:'#c58224',Intermittent:'#4b78ad',Lumpy:'#b95169'};

export function RegimeChart({items,selected,onSelect}) {
  const valid=items.filter(x=>x.ADI>0&&x.CV2>0&&Number.isFinite(x.ADI));
  const xmax=Math.max(1.2,...valid.map(x=>Math.log10(x.ADI)))+.1;
  const ymin=Math.min(-2,...valid.map(x=>Math.log10(x.CV2)))-.1,ymax=Math.max(1,...valid.map(x=>Math.log10(x.CV2)))+.2;
  const X=v=>60+Math.log10(v)/xmax*430,Y=v=>280-(Math.log10(v)-ymin)/(ymax-ymin)*245;
  return <div className="chart"><svg viewBox="0 0 540 330" role="img" aria-label="Interactive ADI and CV squared regime chart">
    {[1,2,5,10,20,50,100].filter(v=>Math.log10(v)<=xmax).map(v=><g key={v}><line className="gridline" x1={X(v)} x2={X(v)} y1="35" y2="280"/><text x={X(v)} y="299" textAnchor="middle">{v}</text></g>)}
    {[.001,.01,.1,1,10,100].filter(v=>Math.log10(v)>=ymin&&Math.log10(v)<=ymax).map(v=><g key={v}><line className="gridline" x1="60" x2="490" y1={Y(v)} y2={Y(v)}/><text x="52" y={Y(v)+4} textAnchor="end">{v}</text></g>)}
    <line className="threshold" x1={X(1.32)} x2={X(1.32)} y1="35" y2="280"/><line className="threshold" x1="60" x2="490" y1={Y(.49)} y2={Y(.49)}/><text x={X(1.32)+5} y="26">ADI 1.32</text><text x="407" y={Y(.49)-7}>CV² 0.49</text>
    {valid.map(x=><circle key={x.part_id} cx={X(x.ADI)} cy={Y(x.CV2)} r={selected===x.part_id?7:4.5} fill={colors[x.regime]||'#84918b'} stroke={selected===x.part_id?'#142e29':'white'} strokeWidth="1.5" tabIndex="0" role="button" aria-label={`Select ${x.part_id}, ${x.regime}`} onClick={()=>onSelect(x.part_id)} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();onSelect(x.part_id);}}}><title>{x.part_id} · {x.regime} · ADI {x.ADI.toFixed(2)} · CV² {x.CV2.toFixed(2)}</title></circle>)}
    <text x="270" y="325" textAnchor="middle">Average demand interval · logarithmic scale</text><text x="13" y="165" transform="rotate(-90 13 165)" textAnchor="middle">CV² · logarithmic scale</text>
  </svg><div className="legend">{Object.entries(colors).map(([name,color])=><span key={name}><i style={{background:color}}/>{name}</span>)}</div>{valid.length<items.length&&<p className="note">{items.length-valid.length} SKUs have zero or undefined values and cannot appear on logarithmic axes.</p>}</div>;
}

export function ForecastChart({item,forecastStart}) {
  if(!item?.forecast)return <p>Select a SKU to inspect its forecast.</p>;
  const history=item.history?.actual||[],f=item.forecast,n=history.length+f.mean.length;
  const max=Math.max(1,...history,...f.q90,...f.mean)*1.12;
  const X=i=>55+i/Math.max(1,n-1)*665,Y=v=>270-v/max*230;
  const line=(values,offset)=>values.map((v,i)=>`${i?'L':'M'}${X(i+offset)},${Y(v)}`).join(' ');
  const band=(low,high)=>`${line(high,history.length)} ${[...low].reverse().map((v,i)=>`L${X(n-1-i)},${Y(v)}`).join(' ')} Z`;
  const dates=[...(item.history?.dates||[]),...f.mean.map((_,i)=>{const d=new Date(forecastStart+'T00:00:00Z');d.setUTCDate(d.getUTCDate()+i);return d.toISOString().slice(0,10);})];
  return <div className="chart"><svg viewBox="0 0 750 325" role="img" aria-label={`Demand history and forecast quantile fan for ${item.part_id}`}>
    {[0,.25,.5,.75,1].map(v=><g key={v}><line className="gridline" x1="55" x2="720" y1={Y(max*v)} y2={Y(max*v)}/><text x="47" y={Y(max*v)+4} textAnchor="end">{(max*v).toFixed(0)}</text></g>)}
    <path d={band(f.q10,f.q90)} fill="#b7dcd5"/><path d={band(f.q50,f.q80)} fill="#75b5a7"/><path d={line(history,0)} className="actual-line"/><path d={line(f.q50,history.length)} className="median-line"/><path d={line(f.mean,history.length)} className="mean-line"/>
    <line className="threshold" x1={X(history.length)} x2={X(history.length)} y1="30" y2="270"/><text x={Math.min(X(history.length)+8,640)} y="22">Forecast starts</text>
    {[...new Set([0,Math.max(0,history.length-1),n-1])].map(i=><text key={i} x={X(i)} y="294" textAnchor={i===0?'start':i===n-1?'end':'middle'}>{dates[i]}</text>)}<text x="15" y="165" transform="rotate(-90 15 165)" textAnchor="middle">Demand units / day</text>
  </svg><div className="legend"><span>Actual — dark</span><span>Median — green</span><span>Mean / point — amber</span><span>q10–q90 — shaded</span></div></div>;
}
