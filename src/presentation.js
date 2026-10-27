const slides=[...document.querySelectorAll('.slide')],deck=document.getElementById('deck');
let i=Math.max(0,Math.min(slides.length-1,(parseInt(location.hash.slice(1))||1)-1));
function show(n){
  i=Math.max(0,Math.min(slides.length-1,n));
  slides.forEach((s,k)=>s.classList.toggle('on',k===i));
  document.getElementById('cnt').textContent=(i+1)+' / '+slides.length;
  document.getElementById('bar').style.width=((i+1)/slides.length*100)+'%';
  history.replaceState(null,'','#'+(i+1));
}
function fit(){const s=Math.min(innerWidth/1280,innerHeight/720);deck.style.transform='scale('+s+')'}
addEventListener('resize',fit);fit();show(i);
addEventListener('keydown',e=>{
  if(['ArrowRight','ArrowDown',' ','PageDown'].includes(e.key)){e.preventDefault();show(i+1)}
  else if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key)){e.preventDefault();show(i-1)}
  else if(e.key==='Home')show(0);else if(e.key==='End')show(slides.length-1);
  else if(e.key==='f'||e.key==='F'){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen()}
});
addEventListener('click',e=>{show(e.clientX>innerWidth/2?i+1:i-1)});
let x0=null;addEventListener('touchstart',e=>x0=e.touches[0].clientX);
addEventListener('touchend',e=>{if(x0===null)return;const d=e.changedTouches[0].clientX-x0;if(Math.abs(d)>50)show(d<0?i+1:i-1);x0=null});

/* pipeline animation */
(function(){
  const $=id=>document.getElementById(id), slide=$('pipe');
  let mode='flex',t=0,timer=0,alert=false,last=performance.now(),hist=[],prev=0;
  const T=22, rad=d=>d*Math.PI/180;
  const ramp=(t)=>t<3?0:t<9?(t-3)/6:t<16?1:t<17?1-(t-16):0;
  function setMode(m){mode=m;t=0;timer=0;alert=false;hist=[];
    document.querySelectorAll('.m').forEach(b=>b.classList.toggle('on',b.dataset.mode===m))}
  document.querySelectorAll('.m').forEach(b=>b.addEventListener('click',e=>{e.stopPropagation();b.blur();setMode(b.dataset.mode)}));
  function frame(now){
    requestAnimationFrame(frame);
    const dt=Math.min(.05,(now-last)/1000);last=now;
    if(!slide.classList.contains('on'))return;
    t=(t+dt)%T; if(t<dt){timer=0;alert=false;hist=[]}
    const r=ramp(t), n=Math.sin(t*5)*.8;
    // pelvis tilt p, lumbar flexion f (degrees)
    const p=mode==='hinge'?r*38+n:2+n*.3, f=mode==='hinge'?3+n*.4:r*34+n;
    const up=p+f, dr=(r-prev)/dt; prev=r;
    // stage 1
    const P0=[50,128],P1=[P0[0]+34*Math.sin(rad(p)),P0[1]-34*Math.cos(rad(p))];
    const P2=[P1[0]+70*Math.sin(rad(up)),P1[1]-70*Math.cos(rad(up))];
    $('sp').setAttribute('d',`M${P0} L${P1} L${P2}`);
    const m1=[(P0[0]+P1[0])/2,(P0[1]+P1[1])/2];
    $('s1').setAttribute('transform',`translate(${m1}) rotate(${p})`);
    $('s2').setAttribute('transform',`translate(${P2}) rotate(${up})`);
    $('t1').textContent='위 '+up.toFixed(0)+'°';$('t2').textContent='아래 '+p.toFixed(0)+'°';
    // stage 2
    hist.push(f);if(hist.length>80)hist.shift();
    $('hist').setAttribute('points',hist.map((v,i)=>`${10+i*1.78},${140-Math.max(0,Math.min(45,v))*2.6}`).join(' '));
    $('angv').textContent=Math.max(0,f).toFixed(0)+'°';
    $('one').textContent=(up>25&&f<18)?'센서 1개였다면 → 경고 (오알림)':'';
    // stage 3
    const prob=1/(1+Math.exp(-(f-18)/3)), moving=Math.abs(dr)>.5;
    $('act').textContent=moving&&t>15.9&&t<17.1?'전환 중':'정지';
    $('pf').style.width=(prob*100)+'%';$('pfv').textContent=(prob*100).toFixed(0)+'%';
    // stage 4/5
    const pass=prob>.5;
    $('gate').textContent=pass?'감시 대상 → 통과':'기준 미만';
    $('gate').classList.toggle('pass',pass);$('r1').classList.toggle('lit',pass);
    timer=pass?Math.min(5,timer+dt):0;
    if(timer>=5)alert=true;if(!pass)alert=false;
    $('ring').setAttribute('stroke-dashoffset',289*(1-timer/5));
    $('ring').setAttribute('stroke',alert?'#e4572e':'#1b998b');
    $('tmv').textContent=timer.toFixed(1);
    const st=slide.querySelectorAll('.st');
    st[3].classList.toggle('hot',pass);st[4].classList.toggle('hot',alert);
    $('toast').classList.toggle('show',alert);
    let c;
    if(mode==='hinge'&&r>.3)c='두 센서가 함께 기움 → 굴곡각은 작음 → 알림 없음 (센서 1개였다면 오알림)';
    else if(alert)c='5초 이상 지속 → 알림 발송';
    else if(pass)c='기준 초과 — 5초 이어지는지 확인 중';
    else if(r>.05&&r<1&&t<16)c='허리가 말리는 중 — 아직 기준 미만';
    else if(t>=16&&t<18.5)c='자세 회복 → 알림 해제';
    else c='자세 정상 → 아무 일도 없음';
    $('cap').textContent=c;
  }
  requestAnimationFrame(frame);
})();
