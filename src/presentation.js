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
