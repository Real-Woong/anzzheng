(function(){
const NS='http://www.w3.org/2000/svg';
const $=id=>document.getElementById(id);
const el=(tag,attrs,parent)=>{const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(parent)parent.appendChild(e);return e};
const txt=(t,attrs,parent)=>{const e=el('text',attrs,parent);e.textContent=t;return e};

const STEPS=[
 {tag:'1 · 데이터',h:'두 갈래 데이터를 모은다',pts:['시뮬레이터: 센서가 밀리는 상황까지 만들 수 있는 합성 사람 (실험실 32명 + 하루 생활 160명)','HARTH: 실제 사람 22명의 활동 데이터 (사전학습용)','평가용 사람은 학습과 겹치지 않게 따로 둔다'],code:'ml/sim/ · ml/repo/harth_repo.py · session_repo.py'},
 {tag:'2 · 신호처리',h:'센서 값을 의미 있는 채널로',pts:['원시 12채널(IMU 2개 × 6축)을 단위 변환·상보 필터로 정리','굴곡각 = 위 기울기 − 아래 기울기 − 중립 보정값','서기·힙힌지·앉기·구부정 4동작으로 개인 기준을 보정','결과: 모델 입력 8채널'],code:'ml/signal/process.py · ml/config.py (CHANNELS)'},
 {tag:'3 · 윈도우·라벨',h:'2초씩 잘라 정답을 붙인다',pts:['2초(100샘플) 창을 0.5초 stride로 슬라이딩','과제 ID가 곧 라벨: 정지/이동/전환 + 허리 굽힘·몸통 기울기','이동·전환 구간의 자세 라벨은 −1 → 손실에서 제외','정규화 통계는 학습 세트에서만 계산 (정보 누수 방지)'],code:'ml/services/windowing.py · harth_windowing.py'},
 {tag:'4 · 모델',h:'PostureCNN: 몸통 하나, 머리 둘',pts:['Conv1D 3블록으로 2초 신호를 32차원 요약으로 압축','활동 헤드: 정지·이동·전환 (softmax)','자세 헤드: 허리 굽힘·몸통 기울기 (sigmoid, 멀티라벨)','파라미터 약 9천 개 — 노트북에서 몇 분이면 학습'],code:'ml/models/posture_cnn.py'},
 {tag:'5 · 사전학습',h:'HARTH로 먼저 움직임을 배운다',pts:['활동 헤드만 학습 (자세 라벨은 전부 −1이라 손실 0)','큰 공개 데이터에서 "움직임 일반 상식"을 몸통에 담는다','8 epoch, batch 512'],code:'ml/services/pretraining.py · ml/app/pretrain.py'},
 {tag:'6 · 미세조정',h:'우리 데이터로 이어서, 그리고 비교',pts:['사전학습 가중치에서 출발, 작은 학습률(1e-3)로 15 epoch','비교군: 랜덤에서 시작해 우리 데이터만으로 25 epoch','결과: 정확도 비슷, 학습 시간 22.6분 → 11.6분'],code:'ml/services/trainer.py · ml/app/longrun.py'},
 {tag:'7 · 평가',h:'룰과 모델을 같은 사람들로 채점',pts:['지표: 재현율(진짜 나쁜 자세를 잡은 비율), 시간당 오알림','CNN은 시드 6개 평균. 센서가 밀리는 스트레스 조건도 함께','센서 2개 룰이 이미 강한 기준선 — CNN은 스트레스에서 재현율이 소폭 높음'],code:'ml/services/evaluator.py · ml/app/loso.py'},
 {tag:'8 · 실시간 적용',h:'학습된 모델을 실시간 판정에 쓴다',pts:['저장된 가중치를 LiveMonitor가 불러온다 (버퍼 20초)','같은 윈도우 코드로 0.5초마다 판정','프로필에 맞는 동작이 5초 이어지면 알림'],code:'ml/services/live.py · judge/state_machine.py · notifier.py'}
];

/* 파형 */
function wave(seed,amp,freq,x0,y0,w=230,n=60){let d='';for(let i=0;i<=n;i++){const x=x0+i*w/n,u=i/n*freq;
  const y=y0+amp*(Math.sin(u*2+seed)+.5*Math.sin(u*5.3+seed*2)+.25*Math.sin(u*11+seed*3));d+=(i?'L':'M')+x.toFixed(1)+' '+y.toFixed(1)}return d}
document.querySelectorAll('[data-wave]').forEach(p=>{const a=p.dataset.wave.split(',').map(Number);
  p.setAttribute('d',wave(a[0],a[1],a[2],a[3],a[4],a[5]||230))});

/* scene 2: raw + channels */
(function(){const g=$('raw');for(let i=0;i<12;i++){const y=40+i*23;
  el('path',{d:wave(i+1,5,4+i%3,30,y,200,40),class:(i<6?'c1':'c2')+' draw',pathLength:1000,style:`animation-delay:${i*.08}s;stroke-width:2`},g)}
  const names=['pelvis_rel','lumbar_c','lumbar_norm','trunk_rel','u_gyr_n','l_gyr_n','u_acc_n','l_acc_n'];
  const cg=$('chs');names.forEach((n,i)=>{const y=34+i*37;
    const gg=el('g',{class:'pop',style:`--d:${1.2+i*.18}s`},cg);
    el('rect',{x:560,y,width:210,height:30,rx:9,class:i<4?'bx on':'bx'},gg);
    txt(n,{x:576,y:y+20,class:'s b'},gg)})})();

/* scene 3: cards */
(function(){const g=$('cards');const T=[['정지','정상','t'],['정지','정상','t'],['정지','굽힘','c'],['정지','굽힘','c'],['이동','자세 −1','g'],['이동','자세 −1','g'],['전환','자세 −1','g'],['정지','정상','t'],['정지','굽힘','c'],['정지','정상','t']];
  T.forEach((t,i)=>{const x=30+i*74,gg=el('g',{class:'pop',style:`--d:${.3+i*.35}s`},g);
    const fill={t:'#e3f4f2',c:'#fce9e3',g:'#eceff3'}[t[2]],st={t:'#1b998b',c:'#e4572e',g:'#9fb0c4'}[t[2]];
    el('rect',{x,y:190,width:66,height:84,rx:10,fill,stroke:st,'stroke-width':2},gg);
    txt(t[0],{x:x+33,y:222,'text-anchor':'middle',class:'h2'},gg);
    txt(t[1],{x:x+33,y:248,'text-anchor':'middle',class:'s b',style:'font-size:13px'},gg);
    el('path',{d:wave(i,6,3,x+8,266,50,16),fill:'none',stroke:st,'stroke-width':1.5,style:'transform:translateY(-8px)'},gg)})})();

/* scene 4: model */
(function(){const g=$('blocks');
  const B=[['입력','8 × 100','#eef3f7',110,0],['Conv 16','16 × 100','#fce9e3',110,.6],['MaxPool','16 × 50','#eef3f7',78,1.2],['Conv 32','32 × 50','#fce9e3',96,1.8],['MaxPool','32 × 25','#eef3f7',62,2.4],['Conv 32','32 × 25','#fce9e3',62,3],['평균','32','#e3f4f2',40,3.6]];
  B.forEach((b,i)=>{const x=24+i*96,h=b[3]*1.1,y=130-h/2+20;
    const gg=el('g',{class:'hl',style:`--d:${b[4]}s`},g);
    el('rect',{x,y,width:78,height:h,rx:12,fill:b[2],stroke:i%2?'#e4572e':'#9fb0c4','stroke-width':2},gg);
    txt(b[0],{x:x+39,y:y+h/2,'text-anchor':'middle',class:'h2'},gg);
    txt(b[1],{x:x+39,y:y+h/2+18,'text-anchor':'middle',class:'s',style:'font-size:12px'},gg);
    if(i<B.length-1)el('path',{d:`M${x+80} 150 h14`,class:'ln'},g)});
  const hx=24+7*96-6;
  [['활동 3분류','정지·이동·전환',60,'#1b998b'],['자세 2요소','굽힘·기울기',160,'#e4572e']].forEach((h,i)=>{
    const gg=el('g',{class:'hl',style:`--d:4.2s`},g);
    el('path',{d:`M${hx-14} 150 L${hx} ${h[2]+30}`,class:'ln'},g);
    el('rect',{x:hx,y:h[2],width:96,height:60,rx:12,fill:'#fff',stroke:h[3],'stroke-width':3},gg);
    txt(h[0],{x:hx+48,y:h[2]+26,'text-anchor':'middle',class:'h2'},gg);
    txt(h[1],{x:hx+48,y:h[2]+46,'text-anchor':'middle',class:'s',style:'font-size:12px'},gg)});
  txt('시간축 길이 100 → 25로 줄이면서 채널을 늘려 "신호의 모양"을 요약한다',{x:24,y:280,class:'s'},g);
  txt('몸통(backbone)은 하나, 헤드 둘이 공유 · 파라미터 약 9천 개',{x:24,y:304,class:'s b'},g);
  const dot=el('circle',{r:7,fill:'#e4572e',cx:0,cy:150},g);
  dot.innerHTML='<animate attributeName="cx" values="30;740" dur="6s" repeatCount="indefinite"/><animate attributeName="opacity" values="0;1;1;0" keyTimes="0;.05;.9;1" dur="6s" repeatCount="indefinite"/>'})();

/* scene 7: eval */
const DATA={normal:[['센서 1개 룰',33.3,2.08],['센서 2개 룰',89.8,.02],['CNN 처음부터',91.4,.15],['CNN + HARTH',90.0,.09]],
            stress:[['센서 1개 룰',29.4,2.01],['센서 2개 룰',72.5,.66],['CNN 처음부터',77.4,.74],['CNN + HARTH',75.5,.64]]};
const ev=$('evalbars');
function drawEval(c){ev.innerHTML='';
  txt('재현율 (높을수록 좋음)',{x:20,y:26,class:'s b'},ev);txt('시간당 오알림 (낮을수록 좋음)',{x:420,y:26,class:'s b'},ev);
  DATA[c].forEach((d,i)=>{const y=48+i*68,col=i<2?'#9fb0c4':(i==3?'#1b998b':'#e4572e');
    txt(d[0],{x:20,y,class:'s b'},ev);
    el('rect',{x:20,y:y+8,width:340,height:28,rx:8,class:'trk'},ev);
    const b1=el('rect',{x:20,y:y+8,width:0,height:28,rx:8,fill:i==1?'#1b998b':col,class:'bar'},ev);
    txt(d[1].toFixed(1)+'%',{x:366,y:y+28,class:'s b'},ev);
    txt(d[0],{x:420,y,class:'s b'},ev);
    el('rect',{x:420,y:y+8,width:300,height:28,rx:8,class:'trk'},ev);
    const b2=el('rect',{x:420,y:y+8,width:0,height:28,rx:8,fill:col,class:'bar'},ev);
    txt(d[2].toFixed(2),{x:726,y:y+28,class:'s b'},ev);
    requestAnimationFrame(()=>requestAnimationFrame(()=>{b1.setAttribute('width',340*d[1]/100);b2.setAttribute('width',Math.max(3,300*d[2]/2.2))}))});
  txt('시뮬레이션 데이터 · CNN은 시드 6개 평균 · 같은 평가용 사람들',{x:20,y:330,class:'s dim'},ev)}
document.querySelectorAll('#evalctl button').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('#evalctl button').forEach(x=>x.classList.toggle('on',x===b));drawEval(b.dataset.c)}));

/* scene 8: deploy */
(function(){const g=$('deploy');
  const S=[['학습된 가중치','posture_cnn.pt'],['LiveMonitor','버퍼 20초'],['윈도우 판정','2초 · 0.5초마다'],['상태 머신','프로필 · 5초 지속'],['알림','문장 생성(선택)']];
  S.forEach((s,i)=>{const x=14+i*156,gg=el('g',{class:'pop',style:`--d:${i*.4}s`},g);
    el('rect',{x,y:110,width:130,height:110,rx:16,class:i==4?'bx on':'bx'},gg);
    txt(s[0],{x:x+65,y:156,'text-anchor':'middle',class:'h2'},gg);
    txt(s[1],{x:x+65,y:182,'text-anchor':'middle',class:'s',style:'font-size:12px'},gg);
    txt(i+1,{x:x+65,y:136,'text-anchor':'middle',class:'s b'},gg);
    if(i<4){el('path',{d:`M${x+132} 165 h20`,class:'ln'},g);
      const w=el('g',{transform:`translate(${x+132} 165)`},g);el('circle',{r:6,class:'dot c1 fx',style:`--dx:20px;animation-delay:${i*.3}s`},w)}});
  txt('학습할 때 쓴 윈도우·신호처리 코드를 그대로 재사용 → 평가와 데모가 같은 결과',{x:14,y:290,class:'s b'},g);
  txt('python -m ml.app.record --fake --live --engine cnn',{x:14,y:318,class:'s dim'},g)})();

/* nav + control */
const nav=$('nav');
STEPS.forEach((s,i)=>{const b=document.createElement('button');b.innerHTML=`<b>${i+1}</b>${s.tag.split(' · ')[1]}`;b.onclick=()=>{go(i);restart()};nav.appendChild(b)});
let cur=0,playing=true,tm=null,t0=0;const DUR=11000;
function go(n){cur=(n+STEPS.length)%STEPS.length;const s=STEPS[cur];
  document.querySelectorAll('.scene').forEach((e,i)=>e.classList.toggle('on',i===cur));
  [...nav.children].forEach((b,i)=>{b.classList.toggle('on',i===cur);b.classList.toggle('done',i<cur)});
  nav.children[cur].scrollIntoView({block:'nearest',inline:'center',behavior:'smooth'});
  $('tag').textContent=s.tag;$('h').textContent=s.h;$('code').textContent=s.code;
  $('pts').innerHTML=s.pts.map(p=>`<li>${p}</li>`).join('');
  $('evalctl').hidden=cur!==6;if(cur===6){document.querySelectorAll('#evalctl button')[0].click()}
  history.replaceState(null,'','#'+(cur+1));t0=performance.now()}
function restart(){t0=performance.now()}
function tick(now){requestAnimationFrame(tick);
  if(!playing){$('prog').style.width='0';return}
  const p=(now-t0)/DUR;$('prog').style.width=Math.min(100,p*100)+'%';if(p>=1)go(cur+1)}
$('prev').onclick=()=>{go(cur-1)};$('next').onclick=()=>{go(cur+1)};
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'❚❚':'▶';t0=performance.now()};
addEventListener('keydown',e=>{if(e.key==='ArrowRight'){go(cur+1)}else if(e.key==='ArrowLeft'){go(cur-1)}else if(e.key===' '){e.preventDefault();$('play').click()}});
go(Math.max(0,Math.min(7,(parseInt(location.hash.slice(1))||1)-1)));requestAnimationFrame(tick);
})();
