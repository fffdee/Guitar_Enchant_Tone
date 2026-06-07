--[[
  ESP-GT-XFORM 尤克里里训练数据采集 (Reaper ReaScript / Lua) —— 对应 IMPLEMENTATION.md §1。
  一次性运行, 不依赖常驻 MCP 桥。源乐器=吉他(Ample Guitar M), 目标乐器=尤克里里(Ample Ethno U)。

  关键: 频域谱掩码不能移调, 故源吉他与尤克里里必须弹"相同音高"。尤克里里音域(约C4-A5)
  高于吉他/贝斯, 因此本脚本把共享 MIDI 限制在 吉他∩尤克里里 公共音区 [60,79] = C4..G5,
  保证两路同音高、且尤克里里都在可发声音域内 (做好音域映射, 不会无声/学不动)。

  生成 4 个 clip(编号从 START_CLIP 起, 默认 5, 保留已有 1-4 的 guitar/bass 低音区数据):
    clip_0005 单音·半音上行   clip_0006 单音·长持续上行
    clip_0007 双音音程        clip_0008 和弦/复音
  逐 clip 用"时间选区 + 仅 solo 单轨"渲染严格对齐的两路 WAV:
    dataset/raw/clip_000X/{guitar.wav, ukulele.wav} + meta.json  (48k / 单声道)

  运行前(只需一次):
    - File > Render 把 Format 设为 WAV、关 Normalize、勾 Tail, 关掉对话框 (Reaper 记住)。
    - Ample 关 Amp/Cab/效果(DI 干信号)、关 Humanize / 随机时间; 确认未 bypass、音色库已加载。
  运行: Actions > Show action list > Load... 选本文件 > Run。
]]--

-------------------- 配置 --------------------
local DATASET_DIR = "e:/DeepLearn/ESP_XFORM/dataset/raw"
local GUITAR_FX   = "Ample Guitar M"
local UKULELE_FX  = "Ample Ethno U"     -- Ample Ethno Ukulele
local CLIP_GAP    = 1.0
local TAIL_MS     = 400
local SR          = 48000
local START_CLIP  = 5                   -- 从 clip_0005 开始(保留 1-4 的 bass 数据)
local LO, HI      = 60, 79              -- C4..G5: 吉他∩尤克里里公共音区
local CLIPS = {
  { mode = "mono_asc",  seconds = 16.0 },
  { mode = "mono_sus",  seconds = 16.0 },
  { mode = "intervals", seconds = 12.0 },
  { mode = "chords",    seconds = 12.0 },
}
---------------------------------------------

local VELS = { 70, 90, 110 }
local CHORDS = { {0,4,7},{0,3,7},{0,4,7,10},{0,3,7,10},{0,5,7},{0,7} }
local INTERVALS = { 3, 4, 5, 7, 12 }

local function msg(s) reaper.ShowConsoleMsg(tostring(s) .. "\n") end

-- base64 解码 (仅用于校验渲染格式是否为 WAV)
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

local function insert_note(take, ts, te, pitch, vel)
  local p0 = reaper.MIDI_GetPPQPosFromProjTime(take, ts)
  local p1 = reaper.MIDI_GetPPQPosFromProjTime(take, te)
  reaper.MIDI_InsertNote(take, false, false, p0, p1, 0, pitch, vel, true)
end

-- 在 [t0, t0+seconds) 写入(源吉他与尤克里里同音高), 仅保留 [LO,HI] 内音符。返回结束偏移。
local function gen_clip(take_g, take_u, t0, mode, seconds)
  local t = 0.2
  local function emit(pitches, dur, vel)
    for _, p in ipairs(pitches) do
      if p >= LO and p <= HI then            -- 公共音区映射: 超出尤克里里音域的音直接丢弃(两路一致)
        insert_note(take_g, t0 + t, t0 + t + dur, p, vel)
        insert_note(take_u, t0 + t, t0 + t + dur, p, vel)
      end
    end
  end
  if mode == "mono_asc" or mode == "mono_sus" then
    local dur, gap, step = 0.40, 0.06, 1
    if mode == "mono_sus" then dur, gap, step = 0.90, 0.12, 1 end
    local p, vi = LO, 0
    while p <= HI and t + dur <= seconds do
      emit({ p }, dur, VELS[(vi % #VELS) + 1]); t = t + dur + gap; p = p + step; vi = vi + 1
    end
  elseif mode == "intervals" then
    local root = LO
    while t + 0.7 <= seconds and root <= HI do
      local iv = INTERVALS[((root - LO) % #INTERVALS) + 1]
      emit({ root, root + iv }, 0.6, VELS[(root % #VELS) + 1]); t = t + 0.7; root = root + 2
    end
  else -- chords
    local root, ci = LO, 0
    while t + 0.85 <= seconds and root <= HI - 7 do
      local c = CHORDS[(ci % #CHORDS) + 1]
      local ps = {}; for _, o in ipairs(c) do ps[#ps + 1] = root + o end
      emit(ps, 0.7, VELS[(ci % #VELS) + 1]); t = t + 0.85; root = root + 3; ci = ci + 1
    end
  end
  return t
end

local function solo_only(target, tracks)
  for _, tr in ipairs(tracks) do
    reaper.SetMediaTrackInfo_Value(tr, "I_SOLO", tr == target and 1 or 0)
  end
end

local function ensure_instrument(track, fxname)
  local idx = reaper.TrackFX_GetInstrument(track)
  if idx < 0 then idx = reaper.TrackFX_AddByName(track, fxname, false, -1) end
  if idx >= 0 then reaper.TrackFX_SetEnabled(track, idx, true) end
  return idx
end

-- 清空一条轨上所有 FX(用于把加载失败的 banjo 换成 ukulele)
local function clear_all_fx(track)
  for i = reaper.TrackFX_GetCount(track) - 1, 0, -1 do
    reaper.TrackFX_Delete(track, i)
  end
end

local function clear_track_items(track)
  for i = reaper.CountTrackMediaItems(track) - 1, 0, -1 do
    reaper.DeleteTrackMediaItem(track, reaper.GetTrackMediaItem(track, i))
  end
end

local function write_meta(dir, idx, mode, dur)
  local f = io.open(dir .. "/meta.json", "w")
  if not f then return end
  f:write(string.format(
    '{\n  "sample_rate": %d,\n  "channels": 1,\n  "instruments": ["guitar", "ukulele"],\n'
    .. '  "source": "guitar",\n  "targets": ["ukulele"],\n  "clip": %d,\n  "mode": "%s",\n'
    .. '  "duration_s": %.3f,\n  "pitch_range": [%d, %d],\n  "guitar_fx": "%s",\n  "ukulele_fx": "%s"\n}\n',
    SR, idx, mode, dur, LO, HI, GUITAR_FX, UKULELE_FX))
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

  -- 轨道: track0=guitar(源), 单独的 ukulele 目标轨(复用/新建 track index 2)
  while reaper.CountTracks(0) < 3 do reaper.InsertTrackAtIndex(reaper.CountTracks(0), true) end
  local gtr = reaper.GetTrack(0, 0)
  local utr = reaper.GetTrack(0, 2)
  reaper.GetSetMediaTrackInfo_String(gtr, "P_NAME", "guitar", true)
  reaper.GetSetMediaTrackInfo_String(utr, "P_NAME", "ukulele", true)

  if ensure_instrument(gtr, GUITAR_FX) < 0 then
    reaper.ShowMessageBox("吉他音源缺失: " .. GUITAR_FX, "错误", 0); return
  end
  -- ukulele 轨: 先清掉(可能加载失败的 banjo)再挂 Ample Ethno U
  clear_all_fx(utr)
  if ensure_instrument(utr, UKULELE_FX) < 0 then
    reaper.ShowMessageBox("尤克里里音源缺失: " .. UKULELE_FX
      .. "\n请确认 Ample Ethno U(Ukulele) 已安装且音色库已加载。", "错误", 0); return
  end

  clear_track_items(gtr)
  clear_track_items(utr)

  local slot = 16.0 + CLIP_GAP
  local total = #CLIPS * slot + 1.0
  local item_g = reaper.CreateNewMIDIItemInProj(gtr, 0, total, false)
  local item_u = reaper.CreateNewMIDIItemInProj(utr, 0, total, false)
  local take_g = reaper.GetActiveTake(item_g)
  local take_u = reaper.GetActiveTake(item_u)

  local clip_dur = {}
  for i, c in ipairs(CLIPS) do
    clip_dur[i] = gen_clip(take_g, take_u, (i - 1) * slot, c.mode, c.seconds)
  end
  reaper.MIDI_Sort(take_g); reaper.MIDI_Sort(take_u)

  -- 渲染设置 (48k / 单声道 / 时间选区 / 保留尾音 / 不入工程)
  reaper.GetSetProjectInfo(0, "RENDER_SRATE", SR, true)
  reaper.GetSetProjectInfo(0, "RENDER_CHANNELS", 1, true)
  reaper.GetSetProjectInfo(0, "RENDER_BOUNDSFLAG", 2, true)
  reaper.GetSetProjectInfo(0, "RENDER_TAILFLAG", 0xFF, true)
  reaper.GetSetProjectInfo(0, "RENDER_TAILMS", TAIL_MS, true)
  reaper.GetSetProjectInfo(0, "RENDER_ADDTOPROJ", 0, true)

  local tracks = { gtr, utr }
  for i, c in ipairs(CLIPS) do
    local t0 = (i - 1) * slot
    local t1 = t0 + clip_dur[i] + 0.3
    reaper.GetSet_LoopTimeRange(true, false, t0, t1, false)
    local dir = string.format("%s/clip_%04d", DATASET_DIR, START_CLIP + i - 1)
    reaper.RecursiveCreateDirectory(dir, 0)
    reaper.GetSetProjectInfo_String(0, "RENDER_FILE", dir, true)

    solo_only(gtr, tracks)
    reaper.GetSetProjectInfo_String(0, "RENDER_PATTERN", "guitar", true)
    reaper.Main_OnCommand(41824, 0)

    solo_only(utr, tracks)
    reaper.GetSetProjectInfo_String(0, "RENDER_PATTERN", "ukulele", true)
    reaper.Main_OnCommand(41824, 0)

    write_meta(dir, START_CLIP + i - 1, c.mode, clip_dur[i])
    msg(string.format("clip_%04d  mode=%-9s dur=%.1fs  rendered (guitar+ukulele)",
      START_CLIP + i - 1, c.mode, clip_dur[i]))
  end

  solo_only(nil, tracks)
  reaper.UpdateArrange()
  reaper.ShowMessageBox(
    string.format("完成: %d 个尤克里里 clip 已渲染到\n%s\n(clip_%04d..clip_%04d, 48k/单声道, 音区 %d-%d)\n\n"
      .. "下一步(回到 Cursor):\npython scripts/preprocess_mask.py --config configs/mask.json "
      .. "--instruments configs/instruments.json",
      #CLIPS, DATASET_DIR, START_CLIP, START_CLIP + #CLIPS - 1, LO, HI),
    "尤克里里采集完成", 0)
  msg("[collect_ukulele] done")
end

reaper.Undo_BeginBlock()
main()
reaper.Undo_EndBlock("ESP-GT-XFORM collect ukulele dataset", -1)
