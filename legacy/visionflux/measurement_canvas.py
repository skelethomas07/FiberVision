from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Mapping

import numpy as np
from PIL import Image

_COMPONENT = None


def _as_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return np.zeros(arr.shape, np.uint8)
    lo, hi = np.percentile(finite, [0.5, 99.5])
    return np.round(np.clip((arr - lo) / max(float(hi - lo), 1e-9), 0, 1) * 255).astype(np.uint8)


def image_to_data_url(image: np.ndarray) -> str:
    arr = _as_uint8(image)
    mode = "L" if arr.ndim == 2 else "RGB"
    pil = Image.fromarray(arr if arr.ndim == 2 else arr[..., :3], mode=mode)
    buffer = BytesIO()
    pil.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _normalize_point(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        point = [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None
    return point if all(np.isfinite(point)) else None


def normalize_canvas_payload(payload: Any) -> dict:
    """Validate a browser Apply/Autosave payload.

    Extra metadata is retained only for fields used to associate a manual chord
    with an already detected model path. The server never trusts browser geometry
    without numeric validation.
    """
    if not isinstance(payload, dict):
        return {"new_measurements": [], "delete_ids": [], "canvas_state": {}}
    new_items: list[dict[str, Any]] = []
    for item in payload.get("new_measurements", []) or []:
        if not isinstance(item, dict):
            continue
        p1 = _normalize_point(item.get("p1"))
        p2 = _normalize_point(item.get("p2"))
        if p1 is None or p2 is None:
            continue
        normalized: dict[str, Any] = {"p1": p1, "p2": p2}
        for key in ("fiber_region_id", "fiber_path_id", "replacement_for"):
            value = item.get(key)
            if value not in (None, ""):
                normalized[key] = str(value)
        try:
            direction = float(item.get("direction_deg"))
            if np.isfinite(direction):
                normalized["direction_deg"] = direction
        except (TypeError, ValueError):
            pass
        new_items.append(normalized)
    delete_ids: list[str] = []
    seen: set[str] = set()
    for value in payload.get("delete_ids", []) or []:
        if value is None:
            continue
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            delete_ids.append(text)
    state = payload.get("canvas_state", {})
    if not isinstance(state, dict):
        state = {}
    return {
        "new_measurements": new_items,
        "delete_ids": delete_ids,
        "canvas_state": state,
    }


_HTML = r"""
<div class="vf-shell">
  <div class="vf-toolbar" role="toolbar" aria-label="측정 도구">
    <button data-tool="pan" title="이동·선택: 드래그로 이동하고 두께선을 클릭하면 해당 fiber를 강조합니다.">↔ 이동·선택 <span class="vf-q">?</span></button>
    <button data-tool="add" class="active" title="두께 추가: 첫 edge를 클릭한 뒤 반대쪽 edge를 클릭합니다. 검출된 fiber라면 자동 법선을 안내합니다.">＋ 두께 추가 <span class="vf-q">?</span></button>
    <button data-tool="modify" title="두께 수정: 기존 두께선의 한쪽 끝을 고른 뒤 새 edge 위치를 클릭합니다.">✎ 두께 수정 <span class="vf-q">?</span></button>
    <button data-tool="erase" title="지우개: 삭제할 자동 또는 수동 두께선을 클릭합니다.">⌫ 지우개 <span class="vf-q">?</span></button>
    <span class="vf-divider"></span>
    <button data-action="undo" title="마지막 추가·수정·삭제 작업을 되돌립니다.">↶ 실행 취소</button>
    <button data-action="clear" title="아직 반영하지 않은 추가·수정 두께선만 지웁니다.">임시선 지우기</button>
    <button data-action="clear-auto" title="모델이 자동으로 만든 측정만 모두 삭제 대상으로 표시합니다. 이미 추가한 수동 측정은 유지됩니다.">자동 측정 전체 지우기 <span class="vf-q">?</span></button>
    <button data-action="fit" title="현재 선택한 섹터 또는 전체 이미지에 화면을 맞춥니다.">화면 맞춤</button>
    <button data-action="magnifier" class="active" title="커서를 따라다니는 돋보기를 켜거나 끕니다.">⌕ 돋보기 ON</button>
    <button data-action="labels" class="active" title="화면의 fiber 라벨만 숨깁니다. 데이터 라벨은 유지됩니다."># 라벨 ON</button>
    <span class="vf-divider"></span>
    <label class="vf-sector-label" title="전체 이미지 또는 여러 섹터로 나누어 검토합니다.">검토 보기
      <select data-action="sector-layout">
        <option value="1x1">전체</option>
        <option value="3x2">6개</option>
        <option value="3x3">9개</option>
        <option value="4x3">12개</option>
        <option value="4x4">16개</option>
      </select>
    </label>
    <button data-action="prev-sector" title="이전 섹터로 이동합니다.">이전</button>
    <button data-action="next-sector" title="현재 섹터를 완료 처리하고 다음 섹터로 이동합니다.">완료·다음</button>
    <span class="vf-sector-progress"></span>
    <span class="vf-spacer"></span>
    <button data-action="apply" class="primary" title="추가·수정·삭제한 측정값을 한 번에 최종 결과에 반영합니다.">전체 반영 <span class="vf-q">?</span></button>
  </div>
  <div class="vf-stage" tabindex="0">
    <canvas class="vf-main-canvas"></canvas>
    <canvas class="vf-magnifier" width="220" height="220"></canvas>
    <div class="vf-selection"></div>
    <div class="vf-hint"></div>
  </div>
  <div class="vf-status">
    <span class="vf-mode"></span>
    <span class="vf-counts"></span>
    <span class="vf-save-status"></span>
  </div>
</div>
"""

_CSS = r"""
.vf-shell { width:100%; font-family:var(--st-font, sans-serif); color:var(--st-text-color); }
.vf-toolbar { display:flex; gap:.38rem; align-items:center; flex-wrap:wrap; padding:.55rem .6rem; border:1px solid color-mix(in srgb, var(--st-text-color) 16%, transparent); border-radius:.75rem .75rem 0 0; background:color-mix(in srgb, var(--st-secondary-background-color) 92%, transparent); }
.vf-toolbar button, .vf-toolbar select { border:1px solid color-mix(in srgb, var(--st-text-color) 18%, transparent); background:var(--st-background-color); color:var(--st-text-color); border-radius:.48rem; padding:.40rem .62rem; cursor:pointer; font-size:.84rem; }
.vf-toolbar button:hover { border-color:var(--st-primary-color); }
.vf-toolbar button.active { color:white; background:var(--st-primary-color); border-color:var(--st-primary-color); }
.vf-toolbar button.primary { color:white; background:var(--st-primary-color); border-color:var(--st-primary-color); font-weight:650; }
.vf-toolbar button:disabled { opacity:.45; cursor:not-allowed; }
.vf-q { display:inline-grid; place-items:center; width:1rem; height:1rem; margin-left:.18rem; border-radius:50%; border:1px solid currentColor; font-size:.68rem; opacity:.72; }
.vf-divider { width:1px; height:1.65rem; background:color-mix(in srgb, var(--st-text-color) 18%, transparent); }
.vf-spacer { flex:1; }
.vf-sector-label { display:flex; gap:.35rem; align-items:center; font-size:.82rem; }
.vf-sector-progress { font-size:.78rem; min-width:5.5rem; }
.vf-stage { position:relative; width:100%; height:min(74vh, 800px); min-height:500px; overflow:hidden; background:#0b0d10; border-left:1px solid color-mix(in srgb, var(--st-text-color) 16%, transparent); border-right:1px solid color-mix(in srgb, var(--st-text-color) 16%, transparent); outline:none; }
.vf-main-canvas { display:block; width:100%; height:100%; touch-action:none; }
.vf-magnifier { position:absolute; left:14px; top:14px; width:220px; height:220px; border:2px solid white; border-radius:10px; background:black; box-shadow:0 4px 18px rgba(0,0,0,.52); pointer-events:none; display:none; z-index:4; }
.vf-selection { position:absolute; left:12px; bottom:12px; max-width:360px; background:rgba(0,0,0,.76); color:white; padding:.55rem .7rem; border-radius:.6rem; font-size:.8rem; pointer-events:none; display:none; z-index:3; }
.vf-hint { position:absolute; left:50%; top:.7rem; transform:translateX(-50%); background:rgba(0,0,0,.65); color:white; padding:.34rem .62rem; border-radius:999px; font-size:.78rem; pointer-events:none; opacity:.92; z-index:3; }
.vf-status { display:flex; justify-content:space-between; gap:1rem; padding:.45rem .65rem; border:1px solid color-mix(in srgb, var(--st-text-color) 16%, transparent); border-radius:0 0 .75rem .75rem; color:color-mix(in srgb, var(--st-text-color) 78%, transparent); font-size:.8rem; }
@media (max-width: 760px) { .vf-stage { min-height:390px; height:64vh; } .vf-toolbar button { padding:.38rem .5rem; } .vf-magnifier { width:160px; height:160px; } }
"""

_JS = r"""
export default function(component) {
  const { parentElement, data, setTriggerValue } = component;
  const shell = parentElement.querySelector('.vf-shell');
  const stage = shell.querySelector('.vf-stage');
  const canvas = shell.querySelector('.vf-main-canvas');
  const ctx = canvas.getContext('2d');
  const magnifier = shell.querySelector('.vf-magnifier');
  const mctx = magnifier.getContext('2d');
  const hint = shell.querySelector('.vf-hint');
  const selectionPanel = shell.querySelector('.vf-selection');
  const modeLabel = shell.querySelector('.vf-mode');
  const countsLabel = shell.querySelector('.vf-counts');
  const saveLabel = shell.querySelector('.vf-save-status');
  const sectorProgress = shell.querySelector('.vf-sector-progress');
  const sectorSelect = shell.querySelector('[data-action="sector-layout"]');
  const applyButton = shell.querySelector('[data-action="apply"]');
  const magnifierButton = shell.querySelector('[data-action="magnifier"]');
  const labelsButton = shell.querySelector('[data-action="labels"]');
  const clearAutoButton = shell.querySelector('[data-action="clear-auto"]');
  const DELETE_ALL_AUTO_TOKEN = '__VISIONFLUX_DELETE_ALL_AUTO__';

  if (shell.__vfCleanup) shell.__vfCleanup();

  const image = new Image();
  const committed = Array.isArray(data.lines) ? data.lines : [];
  const analysisScale = Number(data.analysis_scale || 1);
  const nmPerPx = data.nm_per_px == null ? null : Number(data.nm_per_px);
  const revision = Number(data.revision || 0);
  const editable = data.editable !== false;
  const hoverDelay = Math.max(250, Number(data.hover_delay_ms || 1500));
  const storageKey = `visionflux:${String(data.autosave_key || 'default')}`;
  const initialState = data.initial_state && typeof data.initial_state === 'object' ? data.initial_state : {};

  let mode = editable ? 'add' : 'pan';
  let pending = [];
  let deleted = new Set();
  let firstPoint = null;
  let normalGuide = null;
  let modifyTarget = null;
  let history = [];
  let scale = 1, offsetX = 0, offsetY = 0;
  let dragging = false, dragStart = null, dragOrigin = null, moved = false;
  let dpr = Math.max(1, window.devicePixelRatio || 1);
  let selectedId = null;
  let pointerImage = null, pointerScreen = null;
  let magnifierEnabled = initialState.magnifierEnabled !== false;
  let labelsEnabled = initialState.labelsEnabled !== false;
  let sectorLayout = String(initialState.sectorLayout || '1x1');
  let sectorIndex = Number(initialState.sectorIndex || 0);
  let completedSectors = new Set(Array.isArray(initialState.completedSectors) ? initialState.completedSectors.map(Number) : []);
  let hoverTimer = null, hoverAnchor = null;

  function snapshot() {
    history.push({
      pending:structuredClone(pending), deleted:Array.from(deleted),
      firstPoint:firstPoint ? {...firstPoint}:null,
      normalGuide:normalGuide ? structuredClone(normalGuide):null,
      modifyTarget:modifyTarget ? structuredClone(modifyTarget):null,
      selectedId,
    });
    if (history.length > 80) history.shift();
  }
  function restoreLast() {
    const previous = history.pop(); if (!previous) return;
    pending = previous.pending; deleted = new Set(previous.deleted); firstPoint = previous.firstPoint;
    normalGuide = previous.normalGuide; modifyTarget = previous.modifyTarget; selectedId = previous.selectedId;
    draw(); queueSave();
  }
  function canvasState() {
    return {
      revision, pending:structuredClone(pending), delete_ids:Array.from(deleted),
      sectorLayout, sectorIndex, completedSectors:Array.from(completedSectors),
      magnifierEnabled, labelsEnabled, savedAt:Date.now(),
    };
  }
  function applyState(saved, allowEdits=true) {
    if (!saved || typeof saved !== 'object') return;
    if (allowEdits && Number(saved.revision ?? revision) === revision) {
      pending = Array.isArray(saved.pending) ? saved.pending : pending;
      deleted = new Set(Array.isArray(saved.delete_ids) ? saved.delete_ids.map(String) : (Array.isArray(saved.deleted) ? saved.deleted.map(String) : Array.from(deleted)));
    }
    sectorLayout = String(saved.sectorLayout || sectorLayout);
    sectorIndex = Number(saved.sectorIndex ?? sectorIndex);
    completedSectors = new Set(Array.isArray(saved.completedSectors) ? saved.completedSectors.map(Number) : Array.from(completedSectors));
    magnifierEnabled = saved.magnifierEnabled !== false;
    labelsEnabled = saved.labelsEnabled !== false;
  }
  function saveState(sendServer=false) {
    const state = canvasState();
    try {
      localStorage.setItem(storageKey, JSON.stringify(state));
      const now = new Date(); saveLabel.textContent = `임시저장 ${now.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}`;
    } catch (_) { saveLabel.textContent = '브라우저 임시저장 불가'; }
    if (sendServer && editable) {
      setTriggerValue('autosave', {
        new_measurements: pending.map(line => ({
          p1:line.p1,p2:line.p2,fiber_region_id:line.fiber_region_id,
          fiber_path_id:line.fiber_path_id,direction_deg:line.direction_deg,
          replacement_for:line.replacement_for,
        })),
        delete_ids:Array.from(deleted), canvas_state:state,
      });
    }
  }
  let saveTimer = null;
  function queueSave() { clearTimeout(saveTimer); saveTimer = setTimeout(() => saveState(false), 900); }
  function restoreState() {
    applyState(initialState, true);
    try {
      const local = JSON.parse(localStorage.getItem(storageKey) || 'null');
      if (local && Number(local.savedAt || 0) >= Number(initialState.savedAt || 0)) applyState(local, true);
      if (local?.savedAt) saveLabel.textContent = `복원됨 ${new Date(local.savedAt).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}`;
    } catch (_) {}
    sectorSelect.value = sectorLayout;
  }

  function layoutShape() { const [cols,rows] = sectorLayout.split('x').map(Number); return {cols,rows,total:cols*rows}; }
  function sectorBounds() {
    const {cols,rows,total} = layoutShape(); sectorIndex = Math.max(0, Math.min(total-1, sectorIndex));
    const col = sectorIndex % cols, row = Math.floor(sectorIndex / cols);
    const sw = image.naturalWidth/cols, sh = image.naturalHeight/rows;
    return {x:col*sw,y:row*sh,w:sw,h:sh};
  }
  function updateSectorStatus() {
    const {total}=layoutShape();
    sectorProgress.textContent = total===1 ? '전체 보기' : `섹터 ${sectorIndex+1}/${total} · 완료 ${completedSectors.size}`;
    shell.querySelector('[data-action="prev-sector"]').disabled = total===1;
    shell.querySelector('[data-action="next-sector"]').disabled = total===1;
  }

  function resizeCanvas() {
    const rect=stage.getBoundingClientRect(); dpr=Math.max(1,window.devicePixelRatio||1);
    canvas.width=Math.max(1,Math.round(rect.width*dpr)); canvas.height=Math.max(1,Math.round(rect.height*dpr)); draw();
  }
  function fitCurrent() {
    const rect=stage.getBoundingClientRect(); if(!image.naturalWidth||!rect.width) return;
    const b=sectorBounds(); const pad=0.04;
    scale=Math.min(rect.width/(b.w*(1+2*pad)),rect.height/(b.h*(1+2*pad)));
    scale=Math.max(scale,.02);
    offsetX=(rect.width-b.w*scale)/2-b.x*scale;
    offsetY=(rect.height-b.h*scale)/2-b.y*scale;
    draw();
  }
  function imageToScreen(point){return{x:point.x*scale+offsetX,y:point.y*scale+offsetY};}
  function screenToImage(x,y){return{x:Math.max(0,Math.min(image.naturalWidth-1,(x-offsetX)/scale)),y:Math.max(0,Math.min(image.naturalHeight-1,(y-offsetY)/scale))};}
  function linePoints(line){return{p1:line.p1||[line.x1,line.y1],p2:line.p2||[line.x2,line.y2]};}
  function lineDistance(px,py,line){
    const {p1,p2}=linePoints(line); const a=imageToScreen({x:+p1[0],y:+p1[1]}),b=imageToScreen({x:+p2[0],y:+p2[1]});
    const vx=b.x-a.x,vy=b.y-a.y,wx=px-a.x,wy=py-a.y,den=vx*vx+vy*vy;
    const t=den>0?Math.max(0,Math.min(1,(wx*vx+wy*vy)/den)):0;
    return Math.hypot(px-(a.x+t*vx),py-(a.y+t*vy));
  }
  function isRecognized(line){return line.source!=='manual' && Array.isArray(line.path_points) && line.path_points.length>=2;}
  function pathNearest(px,py,line){
    const pts=Array.isArray(line.path_points)?line.path_points:[]; let best=null;
    for(let i=0;i<pts.length-1;i++){
      const a=imageToScreen({x:+pts[i][0],y:+pts[i][1]}), b=imageToScreen({x:+pts[i+1][0],y:+pts[i+1][1]});
      const vx=b.x-a.x,vy=b.y-a.y,wx=px-a.x,wy=py-a.y,den=vx*vx+vy*vy;
      const t=den>0?Math.max(0,Math.min(1,(wx*vx+wy*vy)/den)):0;
      const qx=a.x+t*vx,qy=a.y+t*vy,d=Math.hypot(px-qx,py-qy);
      if(!best||d<best.distance)best={distance:d,index:i,t,screen:{x:qx,y:qy},tangent:{x:vx,y:vy}};
    }
    return best;
  }
  function nearestRecognized(p, threshold=22){
    let best=null;
    committed.forEach(line=>{
      if(!isRecognized(line)||isAutoClearHidden(line))return;
      const hit=pathNearest(p.x,p.y,line); if(hit&&(!best||hit.distance<best.hit.distance))best={line,hit};
    });
    return best&&best.hit.distance<=threshold?best:null;
  }
  function nearestCommitted(p){let best=null;committed.forEach((line,index)=>{if(isAutoClearHidden(line))return;const d=lineDistance(p.x,p.y,line);if(!best||d<best.distance)best={line,index,distance:d};});return best;}
  function widthText(line){
    const {p1,p2}=linePoints(line); const aw=Math.hypot(+p2[0]-+p1[0],+p2[1]-+p1[1]);
    const ow=line.width_original_px!=null?+line.width_original_px:aw/analysisScale;
    return nmPerPx!=null&&Number.isFinite(nmPerPx)?`${(ow*nmPerPx).toFixed(2)} nm`:`${ow.toFixed(2)} px`;
  }
  function lineColor(line){return line.source==='manual'?'#16d9e8':(line.source==='orientation'?'#a6ecff':'#ffd54a');}
  function selectedLine(){return committed.find(line=>String(line.id)===String(selectedId))||null;}
  function isAutoClearHidden(line){return deleted.has(DELETE_ALL_AUTO_TOKEN) && line.source!=='manual';}
  function isLineDeleted(line){
    const ids=Array.isArray(line.erase_ids)?line.erase_ids.map(String):[];
    return ids.some(id=>deleted.has(id)) || isAutoClearHidden(line);
  }

  function drawPathOn(context,line,color,transform,width=4){
    const pts=Array.isArray(line.path_points)?line.path_points:[]; if(pts.length<2)return;
    context.save(); context.strokeStyle=color; context.lineWidth=width; context.shadowColor='rgba(0,0,0,.8)';context.shadowBlur=3;context.beginPath();
    pts.forEach((p,i)=>{const q=transform({x:+p[0],y:+p[1]});if(i===0)context.moveTo(q.x,q.y);else context.lineTo(q.x,q.y);});context.stroke();context.restore();
  }
  function drawPath(line,color){drawPathOn(ctx,line,color,imageToScreen,4);}
  function drawLineOn(context,line,color,lineWidth,transform,dashed=false,label=true,selected=false){
    const {p1,p2}=linePoints(line); const a=transform({x:+p1[0],y:+p1[1]}),b=transform({x:+p2[0],y:+p2[1]});
    context.save();context.strokeStyle=color;context.lineWidth=lineWidth;context.setLineDash(dashed?[7,5]:[]);if(selected){context.shadowColor='#fff';context.shadowBlur=5;}
    context.beginPath();context.moveTo(a.x,a.y);context.lineTo(b.x,b.y);context.stroke();context.fillStyle=color;
    for(const p of[a,b]){context.beginPath();context.arc(p.x,p.y,Math.max(2.4,lineWidth+1),0,Math.PI*2);context.fill();}
    if(label&&labelsEnabled){const text=String(line.label??'');const mx=(a.x+b.x)/2,my=(a.y+b.y)/2;context.font=selected?'bold 14px sans-serif':'bold 11px sans-serif';const tw=context.measureText(text).width;context.fillStyle='rgba(0,0,0,.78)';context.fillRect(mx-tw/2-3,my-18,tw+6,16);context.fillStyle='#fff';context.fillText(text,mx-tw/2,my-6);}
    context.restore();
  }
  function drawLine(line,color,lineWidth,dashed=false,label=true,selected=false){drawLineOn(ctx,line,color,lineWidth,imageToScreen,dashed,label,selected);}
  function drawInfiniteGuide(context,guide,transform,color='#7cf7ff'){
    if(!guide)return;const span=Math.max(image.naturalWidth,image.naturalHeight)*2;
    const a=transform({x:guide.origin.x-guide.ux*span,y:guide.origin.y-guide.uy*span});
    const b=transform({x:guide.origin.x+guide.ux*span,y:guide.origin.y+guide.uy*span});
    context.save();context.strokeStyle=color;context.lineWidth=1.5;context.setLineDash([8,6]);context.beginPath();context.moveTo(a.x,a.y);context.lineTo(b.x,b.y);context.stroke();context.restore();
  }
  function projectToGuide(point,guide){
    if(!guide)return point;const dx=point.x-guide.origin.x,dy=point.y-guide.origin.y,t=dx*guide.ux+dy*guide.uy;
    return{x:guide.origin.x+t*guide.ux,y:guide.origin.y+t*guide.uy};
  }

  function updateSelectionPanel(){
    const line=selectedLine(); if(!line){selectionPanel.style.display='none';return;}
    const direction=line.direction_deg==null?'—':`${Number(line.direction_deg).toFixed(1)}°`;
    selectionPanel.innerHTML=`<strong>Fiber ${line.label??''}</strong><br>두께 ${widthText(line)} · 방향 ${direction}<br>${isRecognized(line)?'모델 경로 인식됨':'수동 측정'}`;
    selectionPanel.style.display='block';
  }
  function placeMagnifier(){
    if(!pointerScreen)return;const rect=stage.getBoundingClientRect();const size=magnifier.getBoundingClientRect().width||220;let left=pointerScreen.x+20,top=pointerScreen.y+20;
    if(left+size+8>rect.width)left=pointerScreen.x-size-20;if(top+size+8>rect.height)top=pointerScreen.y-size-20;
    magnifier.style.left=`${Math.max(8,left)}px`;magnifier.style.top=`${Math.max(8,top)}px`;
  }
  function drawMagnifier(){
    if(!magnifierEnabled||!pointerImage||!pointerScreen||!image.complete){magnifier.style.display='none';return;}
    magnifier.style.display='block';placeMagnifier();
    const sourceSize=58;const sx=Math.max(0,Math.min(image.naturalWidth-sourceSize,pointerImage.x-sourceSize/2));const sy=Math.max(0,Math.min(image.naturalHeight-sourceSize,pointerImage.y-sourceSize/2));
    const factor=magnifier.width/sourceSize;const transform=p=>({x:(p.x-sx)*factor,y:(p.y-sy)*factor});
    mctx.clearRect(0,0,magnifier.width,magnifier.height);mctx.imageSmoothingEnabled=false;mctx.drawImage(image,sx,sy,sourceSize,sourceSize,0,0,magnifier.width,magnifier.height);
    const selected=selectedLine();
    committed.forEach(line=>{
      if(isAutoClearHidden(line))return;
      const isDeleted=isLineDeleted(line);const isSelected=selected&&String(line.id)===String(selected.id);
      if(isSelected&&isRecognized(line))drawPathOn(mctx,line,'#00fff0',transform,5);
      drawLineOn(mctx,line,isDeleted?'#ff4856':lineColor(line),isSelected?4:2.2,transform,isDeleted,false,isSelected);
    });
    pending.forEach(line=>drawLineOn(mctx,line,'#44a3ff',3,transform,true,false,false));
    drawInfiniteGuide(mctx,modifyTarget?modifyTarget.guide:normalGuide,transform,modifyTarget?'#ff78dc':'#7cf7ff');
    if(firstPoint){const p=transform(firstPoint);mctx.fillStyle='#44a3ff';mctx.beginPath();mctx.arc(p.x,p.y,6,0,Math.PI*2);mctx.fill();}
    mctx.strokeStyle='rgba(255,255,255,.95)';mctx.lineWidth=1;mctx.setLineDash([]);mctx.beginPath();mctx.moveTo(magnifier.width/2-13,magnifier.height/2);mctx.lineTo(magnifier.width/2+13,magnifier.height/2);mctx.moveTo(magnifier.width/2,magnifier.height/2-13);mctx.lineTo(magnifier.width/2,magnifier.height/2+13);mctx.stroke();
  }
  function draw(){
    const rect=stage.getBoundingClientRect();ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,rect.width,rect.height);ctx.fillStyle='#0b0d10';ctx.fillRect(0,0,rect.width,rect.height);
    if(image.complete&&image.naturalWidth){ctx.imageSmoothingEnabled=scale<1;ctx.drawImage(image,offsetX,offsetY,image.naturalWidth*scale,image.naturalHeight*scale);}
    const selected=selectedLine();
    for(const line of committed){
      if(isAutoClearHidden(line))continue;
      const isDeleted=isLineDeleted(line);const isSelected=selected&&String(line.id)===String(selected.id);
      ctx.globalAlpha=selected&&!isSelected?.28:1;
      if(isSelected&&isRecognized(line))drawPath(line,'#00fff0');
      drawLine(line,isDeleted?'rgba(255,72,86,.95)':lineColor(line),isSelected?4.2:(line.source==='manual'?2.8:2.2),isDeleted,true,isSelected);
    }
    ctx.globalAlpha=1;pending.forEach((line,i)=>drawLine({...line,label:`+${i+1}`},'#44a3ff',2.8,true,true,false));
    drawInfiniteGuide(ctx,modifyTarget?modifyTarget.guide:normalGuide,imageToScreen,modifyTarget?'#ff78dc':'#7cf7ff');
    if(firstPoint){const p=imageToScreen(firstPoint);ctx.fillStyle='#44a3ff';ctx.beginPath();ctx.arc(p.x,p.y,5,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1.2;ctx.stroke();}
    updateStatus();updateSelectionPanel();drawMagnifier();
  }
  function updateStatus(){
    const labels={
      pan:'이동·선택 · 드래그로 이동, 선 클릭으로 강조',
      add:firstPoint?'반대쪽 edge를 클릭하세요':'edge 위에 1.5초 머물면 검출 경로를 강조합니다',
      modify:modifyTarget?'새 edge 위치를 클릭하세요':'수정할 두께선의 한쪽 끝을 클릭하세요',
      erase:'지울 두께선을 클릭하세요',
    };
    modeLabel.textContent=editable?labels[mode]:'읽기 전용 · 이동과 선택만 가능';hint.textContent=editable?labels[mode]:'다른 작업자가 편집 중입니다';
    const autoClearPending=deleted.has(DELETE_ALL_AUTO_TOKEN);
    countsLabel.textContent=autoClearPending?`임시 측정 ${pending.length}개 · 자동 측정 전체 삭제 예정`:`임시 측정 ${pending.length}개 · 삭제 예정 ${deleted.size}개`;
    clearAutoButton.disabled=!editable||autoClearPending||!committed.some(line=>line.source!=='manual');
    applyButton.disabled=!editable||(pending.length===0&&deleted.size===0);updateSectorStatus();
  }
  function setMode(next){
    if(!editable&&next!=='pan')next='pan';mode=next;firstPoint=null;normalGuide=null;modifyTarget=null;clearHover();
    shell.querySelectorAll('[data-tool]').forEach(btn=>btn.classList.toggle('active',btn.dataset.tool===mode));
    stage.style.cursor=mode==='pan'?'grab':(mode==='erase'?'not-allowed':'crosshair');draw();
  }
  function pointerPosition(event){const rect=canvas.getBoundingClientRect();return{x:event.clientX-rect.left,y:event.clientY-rect.top};}
  function clearHover(){if(hoverTimer){clearTimeout(hoverTimer);hoverTimer=null;}hoverAnchor=null;}
  function scheduleHover(p){
    if(mode!=='add'||dragging||firstPoint)return;
    if(hoverAnchor&&Math.hypot(p.x-hoverAnchor.x,p.y-hoverAnchor.y)<5)return;
    clearHover();hoverAnchor={...p};
    hoverTimer=setTimeout(()=>{
      const hit=nearestRecognized(hoverAnchor,22);selectedId=hit?String(hit.line.id):null;draw();hoverTimer=null;
    },hoverDelay);
  }
  function guideFromRecognized(point,screenPoint){
    const hit=nearestRecognized(screenPoint,26);if(!hit)return null;
    let tx=hit.hit.tangent.x,ty=hit.hit.tangent.y;const len=Math.hypot(tx,ty);if(len<1e-6)return null;tx/=len;ty/=len;
    return{
      origin:{...point},ux:-ty/scale*scale,uy:tx/scale*scale,lineId:String(hit.line.id),
      fiber_region_id:hit.line.fiber_region_id,fiber_path_id:hit.line.fiber_path_id||hit.line.fiber_region_id,
      direction_deg:hit.line.direction_deg,
    };
  }
  function beginModify(p){
    const best=nearestCommitted(p);if(!best||best.distance>18)return;
    const line=best.line,{p1,p2}=linePoints(line);const a=imageToScreen({x:+p1[0],y:+p1[1]}),b=imageToScreen({x:+p2[0],y:+p2[1]});
    const d1=Math.hypot(p.x-a.x,p.y-a.y),d2=Math.hypot(p.x-b.x,p.y-b.y);const endpoint=d1<=d2?'p1':'p2';
    const moving=endpoint==='p1'?{x:+p1[0],y:+p1[1]}:{x:+p2[0],y:+p2[1]},fixed=endpoint==='p1'?{x:+p2[0],y:+p2[1]}:{x:+p1[0],y:+p1[1]};
    let ux=moving.x-fixed.x,uy=moving.y-fixed.y;const len=Math.hypot(ux,uy);if(len<1e-6)return;ux/=len;uy/=len;
    modifyTarget={line,endpoint,fixed,guide:{origin:fixed,ux,uy}};selectedId=String(line.id);draw();
  }

  function onPointerDown(event){stage.focus();const p=pointerPosition(event);moved=false;
    if(mode==='pan'||event.button===1||event.button===2){dragging=true;dragStart=p;dragOrigin={x:offsetX,y:offsetY};stage.setPointerCapture(event.pointerId);stage.style.cursor='grabbing';return;}
    if(event.button!==0||!editable)return;
    if(mode==='add'){
      let point=screenToImage(p.x,p.y);
      if(!firstPoint){snapshot();firstPoint=point;normalGuide=guideFromRecognized(point,p);if(normalGuide)selectedId=normalGuide.lineId;}
      else{
        if(normalGuide)point=projectToGuide(point,normalGuide);const d=Math.hypot(point.x-firstPoint.x,point.y-firstPoint.y);
        if(d>=.75){snapshot();pending.push({
          p1:[firstPoint.x,firstPoint.y],p2:[point.x,point.y],
          fiber_region_id:normalGuide?.fiber_region_id,fiber_path_id:normalGuide?.fiber_path_id,
          direction_deg:normalGuide?.direction_deg,
        });queueSave();}
        firstPoint=null;normalGuide=null;
      }draw();return;
    }
    if(mode==='modify'){
      if(!modifyTarget){beginModify(p);return;}
      snapshot();let point=projectToGuide(screenToImage(p.x,p.y),modifyTarget.guide);const line=modifyTarget.line,{p1,p2}=linePoints(line);
      const nextP1=modifyTarget.endpoint==='p1'?[point.x,point.y]:[+p1[0],+p1[1]];
      const nextP2=modifyTarget.endpoint==='p2'?[point.x,point.y]:[+p2[0],+p2[1]];
      if(Math.hypot(nextP2[0]-nextP1[0],nextP2[1]-nextP1[1])>=.75){
        pending.push({p1:nextP1,p2:nextP2,replacement_for:String(line.id),fiber_region_id:line.fiber_region_id,fiber_path_id:line.fiber_path_id||line.fiber_region_id,direction_deg:line.direction_deg});
        const ids=Array.isArray(line.erase_ids)?line.erase_ids.map(String):[];ids.forEach(id=>deleted.add(id));queueSave();
      }
      modifyTarget=null;draw();return;
    }
    if(mode==='erase'){
      let best=null;pending.forEach((line,index)=>{const d=lineDistance(p.x,p.y,line);if(!best||d<best.distance)best={kind:'pending',index,distance:d};});committed.forEach((line,index)=>{const d=lineDistance(p.x,p.y,line);if(!best||d<best.distance)best={kind:'committed',index,distance:d};});
      if(best&&best.distance<=14){snapshot();if(best.kind==='pending')pending.splice(best.index,1);else{const ids=Array.isArray(committed[best.index].erase_ids)?committed[best.index].erase_ids.map(String):[];const already=ids.length&&ids.every(id=>deleted.has(id));ids.forEach(id=>already?deleted.delete(id):deleted.add(id));}draw();queueSave();}
    }
  }
  function onPointerMove(event){
    const p=pointerPosition(event);pointerScreen=p;pointerImage=screenToImage(p.x,p.y);if(dragging){if(Math.hypot(p.x-dragStart.x,p.y-dragStart.y)>3)moved=true;offsetX=dragOrigin.x+(p.x-dragStart.x);offsetY=dragOrigin.y+(p.y-dragStart.y);}else scheduleHover(p);draw();
  }
  function onPointerUp(event){if(!dragging)return;const p=pointerPosition(event);dragging=false;try{stage.releasePointerCapture(event.pointerId);}catch(_){}stage.style.cursor=mode==='pan'?'grab':(mode==='erase'?'not-allowed':'crosshair');if(mode==='pan'&&!moved){const best=nearestCommitted(p);selectedId=best&&best.distance<=15?String(best.line.id):null;}draw();}
  function onPointerLeave(){pointerImage=null;pointerScreen=null;clearHover();magnifier.style.display='none';}
  function onWheel(event){event.preventDefault();const p=pointerPosition(event),before=screenToImage(p.x,p.y),factor=Math.exp(-event.deltaY*.0015);scale=Math.max(.05,Math.min(28,scale*factor));offsetX=p.x-before.x*scale;offsetY=p.y-before.y*scale;draw();}
  function onKeyDown(event){
    if(event.key==='Escape'){firstPoint=null;normalGuide=null;modifyTarget=null;selectedId=null;draw();}
    if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='z'){event.preventDefault();restoreLast();}
    if(event.key==='1')setMode('pan');if(event.key==='2')setMode('add');if(event.key==='3')setMode('modify');if(event.key==='4')setMode('erase');
  }

  shell.querySelectorAll('[data-tool]').forEach(button=>{button.disabled=!editable&&button.dataset.tool!=='pan';button.onclick=()=>setMode(button.dataset.tool);});
  shell.querySelector('[data-action="undo"]').onclick=restoreLast;
  shell.querySelector('[data-action="clear"]').onclick=()=>{if(!pending.length&&!firstPoint&&!modifyTarget)return;snapshot();pending=[];firstPoint=null;normalGuide=null;modifyTarget=null;draw();queueSave();};
  clearAutoButton.onclick=()=>{
    if(!editable)return;
    const automaticCount=committed.filter(line=>line.source!=='manual').length;
    if(!automaticCount||deleted.has(DELETE_ALL_AUTO_TOKEN))return;
    const confirmed=window.confirm(`현재 표시된 모델 자동 측정 ${automaticCount}개를 모두 지울까요?\n사용자가 추가한 수동 측정은 유지됩니다.`);
    if(!confirmed)return;
    snapshot();deleted.add(DELETE_ALL_AUTO_TOKEN);
    const selected=selectedLine();if(selected&&selected.source!=='manual')selectedId=null;
    firstPoint=null;normalGuide=null;modifyTarget=null;draw();queueSave();
  };
  shell.querySelector('[data-action="fit"]').onclick=fitCurrent;
  magnifierButton.onclick=()=>{magnifierEnabled=!magnifierEnabled;magnifierButton.classList.toggle('active',magnifierEnabled);magnifierButton.textContent=magnifierEnabled?'⌕ 돋보기 ON':'⌕ 돋보기 OFF';draw();queueSave();};
  labelsButton.onclick=()=>{labelsEnabled=!labelsEnabled;labelsButton.classList.toggle('active',labelsEnabled);labelsButton.textContent=labelsEnabled?'# 라벨 ON':'# 라벨 OFF';draw();queueSave();};
  sectorSelect.onchange=()=>{sectorLayout=sectorSelect.value;sectorIndex=0;completedSectors=new Set();fitCurrent();queueSave();};
  shell.querySelector('[data-action="prev-sector"]').onclick=()=>{const {total}=layoutShape();sectorIndex=(sectorIndex-1+total)%total;fitCurrent();queueSave();};
  shell.querySelector('[data-action="next-sector"]').onclick=()=>{const {total}=layoutShape();completedSectors.add(sectorIndex);sectorIndex=(sectorIndex+1)%total;fitCurrent();queueSave();};
  applyButton.onclick=()=>{
    if(!editable||(!pending.length&&!deleted.size))return;saveState(false);applyButton.disabled=true;
    setTriggerValue('apply',{
      new_measurements:pending.map(line=>({p1:line.p1,p2:line.p2,fiber_region_id:line.fiber_region_id,fiber_path_id:line.fiber_path_id,direction_deg:line.direction_deg,replacement_for:line.replacement_for})),
      delete_ids:Array.from(deleted),canvas_state:canvasState(),
    });
  };

  stage.addEventListener('pointerdown',onPointerDown);stage.addEventListener('pointermove',onPointerMove);stage.addEventListener('pointerup',onPointerUp);stage.addEventListener('pointercancel',onPointerUp);stage.addEventListener('pointerleave',onPointerLeave);stage.addEventListener('wheel',onWheel,{passive:false});stage.addEventListener('keydown',onKeyDown);stage.addEventListener('contextmenu',e=>e.preventDefault());
  const observer=new ResizeObserver(resizeCanvas);observer.observe(stage);restoreState();
  const autosaveInterval=setInterval(()=>saveState(true),300000);
  image.onload=()=>{resizeCanvas();fitCurrent();};image.src=String(data.image_url||'');setMode(mode);
  magnifierButton.classList.toggle('active',magnifierEnabled);magnifierButton.textContent=magnifierEnabled?'⌕ 돋보기 ON':'⌕ 돋보기 OFF';
  labelsButton.classList.toggle('active',labelsEnabled);labelsButton.textContent=labelsEnabled?'# 라벨 ON':'# 라벨 OFF';

  const cleanup=()=>{observer.disconnect();clearInterval(autosaveInterval);clearTimeout(saveTimer);clearHover();stage.removeEventListener('pointerdown',onPointerDown);stage.removeEventListener('pointermove',onPointerMove);stage.removeEventListener('pointerup',onPointerUp);stage.removeEventListener('pointercancel',onPointerUp);stage.removeEventListener('pointerleave',onPointerLeave);stage.removeEventListener('wheel',onWheel);stage.removeEventListener('keydown',onKeyDown);};
  shell.__vfCleanup=cleanup;return cleanup;
}
"""


def measurement_canvas(
    image: np.ndarray,
    lines: list[dict],
    analysis_scale: float,
    nm_per_px: float | None,
    revision: int,
    *,
    key: str,
    autosave_key: str | None = None,
    initial_state: Mapping[str, Any] | None = None,
    editable: bool = True,
    hover_delay_ms: int = 1500,
):
    """Render the interactive review component.

    Latency-sensitive interactions stay in the browser. Python reruns after Apply
    or after the five-minute autosave event used by optional Supabase sharing.
    """
    import streamlit as st

    global _COMPONENT
    if _COMPONENT is None:
        _COMPONENT = st.components.v2.component(
            "visionflux.measurement_canvas",
            html=_HTML,
            css=_CSS,
            js=_JS,
            isolate_styles=True,
        )
    return _COMPONENT(
        data={
            "image_url": image_to_data_url(image),
            "lines": lines,
            "analysis_scale": float(analysis_scale),
            "nm_per_px": None if nm_per_px is None else float(nm_per_px),
            "revision": int(revision),
            "autosave_key": str(autosave_key or key),
            "initial_state": dict(initial_state or {}),
            "editable": bool(editable),
            "hover_delay_ms": int(hover_delay_ms),
        },
        key=key,
        on_apply_change=lambda: None,
        on_autosave_change=lambda: None,
    )
