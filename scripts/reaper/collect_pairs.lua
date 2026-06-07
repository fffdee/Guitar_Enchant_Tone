--[[
  ESP-GT-XFORM 配对轨数据采集 (Reaper ReaScript / Lua) —— 适配"源/目标分轨 + 逐 item 切分"工程。
  ------------------------------------------------------------------
  适配你当前的工程结构：每个目标乐器都有一条【单独准备好的吉他源轨】与之配对，
  且把演奏切成了一个个【MIDI item】，每个 item 就是一个 clip：

      guitar_bass    (源) ↔ bass_data    (目标 bass)
      guitar_ukulele (源) ↔ ukulele_data (目标 ukulele)

  本脚本【不镜像、不按间隔切】，而是：对每一对轨，按【item 索引】逐段渲染——
  用每个 item 的 [位置, 位置+长度] 作为时间选区，solo 源轨渲染 guitar.wav、
  solo 目标轨渲染 <target>.wav，二者同段对齐。源/目标 item 需位置/长度一一对应
  (你的工程已满足)。输出到新数据集，旧 dataset/raw 不动也不用：

      dataset/raw_style/clip_XXXX/{guitar.wav, <target>.wav, meta.json}  (48k/单声道)

  多次运行自动接着已有最大编号追加。

  注意(频域谱掩码不能移调)：每对的【源吉他与目标必须同音高】。本脚本忠实渲染你写的音符，
  不做任何移调；若某对源/目标音高不一致，训练出的该音色会偏弱(见运行后控制台的音高校验提示)。

  运行前(只需一次)：File > Render：Format=WAV、关 Normalize、勾 Tail，关掉对话框。
  运行：Actions > Show action list > Load... 选本文件 > Run。
]]--

-------------------- 配置 --------------------
local DATASET_DIR = "e:/DeepLearn/ESP_XFORM/dataset/raw_style"
-- 源/目标轨配对：{ 源轨名, 目标轨名, 目标乐器名(=输出 wav 名/训练乐器名), 源音源, 目标音源 }
local PAIRS = {
  { src = "guitar_bass",    tgt = "bass_data",    name = "bass",    src_fx = "Ample Guitar M", tgt_fx = "Ample Bass P" },
  { src = "guitar_ukulele", tgt = "ukulele_data", name = "ukulele", src_fx = "Ample Guitar M", tgt_fx = "Ample Ethno U" },
}
local PRE_ROLL = 0.02      -- item 起点前留白(秒)
local TAIL_S   = 0.40      -- 末尾保留尾音(秒)
local SR       = 48000
---------------------------------------------

local function msg(s) reaper.ShowConsoleMsg(tostring(s) .. "\n") end

-- base64 解码 (校验渲染格式是否为 WAV)
local B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
local function b64dec(data)
  data = string.gsub(data, '[^' .. B64 .. '=]', '')
  return (data:gsub('.', function(x)
    if x == '=' then return '' end
    local r, f = '', (B64:find(x) - 1)
    for i = 6, 1, -1 do r = r .. (f % 2 ^ i - f % 2 ^ (i - 1) > 0 and '1' or '0') end
    return r
  end):gsub('%d%d%d?%d?%d?%d?%d?%d?', function(x)
    if #x ~= 8 then return '' end
    local c = 0
    for i = 1, 8 do c = c + (x:sub(i, i) == '1' and 2 ^ (8 - i) or 0) end
    return string.char(c)
  end))
end
local function is_wav_format()
  local _, fmt = reaper.GetSetProjectInfo_String(0, "RENDER_FORMAT", "", false)
  if not fmt or #fmt < 4 then return false end
  local raw = b64dec(fmt)
  return #raw >= 4 and raw:sub(1, 4):reverse() == "wave"
end

local function find_track_by_name(name)
  for i = 0, reaper.CountTracks(0) - 1 do
    local tr = reaper.GetTrack(0, i)
    local _, nm = reaper.GetSetMediaTrackInfo_String(tr, "P_NAME", "", false)
    if nm == name then return tr end
  end
  return nil
end

local function ensure_instrument(track, fxname)
  local idx = reaper.TrackFX_GetInstrument(track)
  if idx < 0 then idx = reaper.TrackFX_AddByName(track, fxname, false, -1) end
  if idx >= 0 then reaper.TrackFX_SetEnabled(track, idx, true) end
  return idx
end

local function solo_only(target, tracks)
  for _, tr in ipairs(tracks) do
    reaper.SetMediaTrackInfo_Value(tr, "I_SOLO", tr == target and 1 or 0)
  end
end

-- item 的首音高(用于源/目标同音高粗校验)
local function first_pitch(item)
  local take = reaper.GetActiveTake(item)
  if not take or not reaper.TakeIsMIDI(take) then return nil end
  local ok, _, _, _, _, _, pitch = reaper.MIDI_GetNote(take, 0)
  if ok then return pitch end
  return nil
end

-- 扫描数据集已有 clip_#### 的最大编号(跨多次运行追加)
local function next_clip_index(dir)
  reaper.RecursiveCreateDirectory(dir, 0)
  local maxn, i = 0, 0
  while true do
    local sub = reaper.EnumerateSubdirectories(dir, i)
    if not sub then break end
    local num = string.match(sub, "^clip_(%d+)$")
    if num then maxn = math.max(maxn, tonumber(num)) end
    i = i + 1
  end
  return maxn + 1
end

local function write_meta(dir, idx, name, src_fx, tgt_fx, dur, pitch_off)
  local f = io.open(dir .. "/meta.json", "w")
  if not f then return end
  f:write(string.format(
    '{\n  "sample_rate": %d,\n  "channels": 1,\n  "instruments": ["guitar", "%s"],\n'
    .. '  "source": "guitar",\n  "targets": ["%s"],\n  "clip": %d,\n  "mode": "pairs",\n'
    .. '  "duration_s": %.3f,\n  "src_minus_tgt_first_pitch": %s,\n'
    .. '  "guitar_fx": "%s",\n  "%s_fx": "%s"\n}\n',
    SR, name, name, idx, dur, tostring(pitch_off), src_fx, name, tgt_fx))
  f:close()
end

-------------------- 主流程 --------------------
local function main()
  reaper.ShowConsoleMsg("")
  reaper.GetSetProjectInfo(0, "PROJECT_SRATE", SR, true)
  reaper.GetSetProjectInfo(0, "PROJECT_SRATE_USE", 1, true)

  if not is_wav_format() then
    reaper.ShowMessageBox("渲染格式不是 WAV。请先 File > Render 把 Format 设为 WAV(关 Normalize、勾 Tail)，"
      .. "关掉对话框后重跑本脚本。", "需要 WAV 渲染格式", 0)
    return
  end

  -- 解析配对轨 + 收集所有轨(solo 互斥)
  local jobs, all_tracks = {}, {}
  for _, p in ipairs(PAIRS) do
    local str = find_track_by_name(p.src)
    local ttr = find_track_by_name(p.tgt)
    if not str then
      msg(string.format("跳过 %s：找不到源轨 '%s'", p.name, p.src))
    elseif not ttr then
      msg(string.format("跳过 %s：找不到目标轨 '%s'", p.name, p.tgt))
    else
      ensure_instrument(str, p.src_fx)
      if ensure_instrument(ttr, p.tgt_fx) < 0 then
        msg(string.format("跳过 %s：目标音源 %s 加载失败", p.name, p.tgt_fx))
      else
        jobs[#jobs + 1] = { p = p, str = str, ttr = ttr }
        all_tracks[#all_tracks + 1] = str
        all_tracks[#all_tracks + 1] = ttr
      end
    end
  end
  if #jobs == 0 then
    reaper.ShowMessageBox("没有可用的源/目标轨对。请检查轨名是否为：\n"
      .. "guitar_bass / bass_data / guitar_ukulele / ukulele_data", "无可采集数据", 0)
    return
  end

  -- 渲染设置 (48k/单声道/时间选区/保留尾音/不入工程)
  reaper.GetSetProjectInfo(0, "RENDER_SRATE", SR, true)
  reaper.GetSetProjectInfo(0, "RENDER_CHANNELS", 1, true)
  reaper.GetSetProjectInfo(0, "RENDER_BOUNDSFLAG", 2, true)
  reaper.GetSetProjectInfo(0, "RENDER_TAILFLAG", 0xFF, true)
  reaper.GetSetProjectInfo(0, "RENDER_TAILMS", math.floor(TAIL_S * 1000), true)
  reaper.GetSetProjectInfo(0, "RENDER_ADDTOPROJ", 0, true)

  local clip_idx = next_clip_index(DATASET_DIR)
  local total = 0

  for _, job in ipairs(jobs) do
    local p, str, ttr = job.p, job.str, job.ttr
    local ns = reaper.CountTrackMediaItems(str)
    local nt = reaper.CountTrackMediaItems(ttr)
    local n = math.min(ns, nt)
    if ns ~= nt then
      msg(string.format("[警告] %s: 源 item 数(%d) 与目标(%d) 不一致，按较小值 %d 配对。",
        p.name, ns, nt, n))
    end
    msg(string.format("== %s: %d 段 ==", p.name, n))

    for i = 0, n - 1 do
      local sitem = reaper.GetTrackMediaItem(str, i)
      local titem = reaper.GetTrackMediaItem(ttr, i)
      local pos = reaper.GetMediaItemInfo_Value(sitem, "D_POSITION")
      local len = reaper.GetMediaItemInfo_Value(sitem, "D_LENGTH")

      -- 源/目标首音高差(同音高应为 0)
      local sp, tp = first_pitch(sitem), first_pitch(titem)
      local off = (sp and tp) and (sp - tp) or "null"

      local t0 = math.max(0.0, pos - PRE_ROLL)
      local t1 = pos + len
      reaper.GetSet_LoopTimeRange(true, false, t0, t1, false)

      local dir = string.format("%s/clip_%04d", DATASET_DIR, clip_idx)
      reaper.RecursiveCreateDirectory(dir, 0)
      reaper.GetSetProjectInfo_String(0, "RENDER_FILE", dir, true)

      solo_only(str, all_tracks)
      reaper.GetSetProjectInfo_String(0, "RENDER_PATTERN", "guitar", true)
      reaper.Main_OnCommand(41824, 0)

      solo_only(ttr, all_tracks)
      reaper.GetSetProjectInfo_String(0, "RENDER_PATTERN", p.name, true)
      reaper.Main_OnCommand(41824, 0)

      write_meta(dir, clip_idx, p.name, p.src_fx, p.tgt_fx, len, off)
      msg(string.format("  clip_%04d  %-8s  seg %d  %.2f..%.2fs  pitchΔ(src-tgt)=%s",
        clip_idx, p.name, i, pos, pos + len, tostring(off)))
      clip_idx = clip_idx + 1
      total = total + 1
    end
  end

  solo_only(nil, all_tracks)
  reaper.UpdateArrange()
  reaper.ShowMessageBox(
    string.format("完成：新增 %d 个 clip 到\n%s\n(48k/单声道, 逐 item 配对渲染)\n\n"
      .. "若控制台出现 pitchΔ != 0，说明该对源/目标不同音高(谱掩码学不准)，建议改 MIDI 使其一致。\n\n"
      .. "下一步(回到 Cursor)：\npython scripts/preprocess_mask.py --config configs/mask_style.json "
      .. "--instruments configs/instruments.json",
      total, DATASET_DIR),
    "配对数据采集完成", 0)
  msg("[collect_pairs] done, +" .. total .. " clips")
end

reaper.Undo_BeginBlock()
main()
reaper.Undo_EndBlock("ESP-GT-XFORM collect pairs dataset", -1)
