--[[
  ESP-GT-XFORM 训练数据采集 (Reaper ReaScript / Lua) —— 对应 IMPLEMENTATION.md §1。
  一次性运行, 不依赖常驻 MCP 桥。复用工程已有的 guitar/bass 轨与 Ample 音源(若无则加载)。

  生成 4 类 clip 并逐 clip 用"时间选区 + 仅 solo 单轨"渲染严格对齐的双路 WAV:
    clip_0001 单音·半音上行 (低->高)   clip_0002 单音·长持续上行
    clip_0003 双音音程                  clip_0004 和弦/复音
  输出: dataset/raw/clip_XXXX/{guitar.wav, bass.wav} + meta.json  (48k / 单声道)

  运行前(只需一次):
    - File > Render 把 Format 设为 WAV、关 Normalize、勾 Tail, 关掉对话框 (Reaper 记住)。
    - Ample 关 Amp/Cab/效果(DI 干信号)、关 Humanize / 随机时间; 确认未 bypass。
  运行: Actions > Show action list > Load... 选本文件 > Run。
]]--

-------------------- 配置 --------------------
local DATASET_DIR = "e:/DeepLearn/ESP_XFORM/dataset/raw"
local GUITAR_FX   = "Ample Guitar M"
local BASS_FX     = "Ample Bass P"
local CLIP_GAP    = 1.0     -- clip 间在时间轴上的间隔
local TAIL_MS     = 400
local SR          = 48000
-- clip 定义: mode + 该 clip 时长上限(秒)
local CLIPS = {
  { mode = "mono_asc",  seconds = 16.0 },
  { mode = "mono_sus",  seconds = 16.0 },
  { mode = "intervals", seconds = 12.0 },
  { mode = "chords",    seconds = 10.0 },
}
---------------------------------------------

local LO, HI = 40, 76          -- E2..E5
local VELS = { 70, 90, 110 }
local CHORDS = { {0,4,7},{0,3,7},{0,4,7,10},{0,3,7,10},{0,5,7},{0,7} }
local INTERVALS = { 3, 4, 5, 7, 12 }

local function msg(s) reaper.ShowConsoleMsg(tostring(s) .. "\n") end

-- base64 解码 (仅用于校验渲染格式)
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

-- 各模式生成器: 在 [t0, t0+seconds) 写入, 返回实际结束偏移(相对 t0)
local function gen_clip(take_g, take_b, t0, mode, seconds)
  local t = 0.2
  local function emit(pitches, dur, vel)
    for _, p in ipairs(pitches) do
      if p >= 28 and p <= 96 then
        insert_note(take_g, t0 + t, t0 + t + dur, p, vel)
        insert_note(take_b, t0 + t, t0 + t + dur, p, vel)
      end
    end
  end
  if mode == "mono_asc" or mode == "mono_sus" then
    local dur, gap, step = 0.40, 0.06, 1
    if mode == "mono_sus" then dur, gap, step = 0.90, 0.12, 2 end
    local p, vi = LO, 0
    while p <= HI and t + dur <= seconds do
      emit({ p }, dur, VELS[(vi % #VELS) + 1]); t = t + dur + gap; p = p + step; vi = vi + 1
    end
  elseif mode == "intervals" then
    local root = LO
    while t + 0.7 <= seconds and root <= HI do
      local iv = INTERVALS[((root - LO) % #INTERVALS) + 1]
      emit({ root, root + iv }, 0.6, VELS[(root % #VELS) + 1]); t = t + 0.7; root = root + 3
    end
  else -- chords
    local root, ci = LO, 0
    while t + 0.85 <= seconds and root <= HI - 7 do
      local c = CHORDS[(ci % #CHORDS) + 1]
      local ps = {}; for _, o in ipairs(c) do ps[#ps + 1] = root + o end
      emit(ps, 0.7, VELS[(ci % #VELS) + 1]); t = t + 0.85; root = root + 4; ci = ci + 1
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
  if idx >= 0 then reaper.TrackFX_SetEnabled(track, idx, true) end   -- 防 bypass 静音
  return idx
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
    '{\n  "sample_rate": %d,\n  "channels": 1,\n  "instruments": ["guitar", "bass"],\n'
    .. '  "source": "guitar",\n  "targets": ["bass"],\n  "clip": %d,\n  "mode": "%s",\n'
    .. '  "duration_s": %.3f,\n  "guitar_fx": "%s",\n  "bass_fx": "%s"\n}\n',
    SR, idx, mode, dur, GUITAR_FX, BASS_FX))
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

  while reaper.CountTracks(0) < 2 do reaper.InsertTrackAtIndex(reaper.CountTracks(0), true) end
  local gtr = reaper.GetTrack(0, 0)
  local btr = reaper.GetTrack(0, 1)
  reaper.GetSetMediaTrackInfo_String(gtr, "P_NAME", "guitar", true)
  reaper.GetSetMediaTrackInfo_String(btr, "P_NAME", "bass", true)
  if ensure_instrument(gtr, GUITAR_FX) < 0 then reaper.ShowMessageBox("吉他音源缺失: " .. GUITAR_FX, "错误", 0); return end
  if ensure_instrument(btr, BASS_FX) < 0 then reaper.ShowMessageBox("贝斯音源缺失: " .. BASS_FX, "错误", 0); return end

  -- 清空两轨已有 item (避免与旧 MIDI 叠加)
  clear_track_items(gtr)
  clear_track_items(btr)

  -- 时间轴整体 MIDI item (覆盖所有 clip)
  local slot = 16.0 + CLIP_GAP
  local total = #CLIPS * slot + 1.0
  local item_g = reaper.CreateNewMIDIItemInProj(gtr, 0, total, false)
  local item_b = reaper.CreateNewMIDIItemInProj(btr, 0, total, false)
  local take_g = reaper.GetActiveTake(item_g)
  local take_b = reaper.GetActiveTake(item_b)

  local clip_dur = {}
  for i, c in ipairs(CLIPS) do
    clip_dur[i] = gen_clip(take_g, take_b, (i - 1) * slot, c.mode, c.seconds)
  end
  reaper.MIDI_Sort(take_g); reaper.MIDI_Sort(take_b)

  -- 渲染设置
  reaper.GetSetProjectInfo(0, "RENDER_SRATE", SR, true)
  reaper.GetSetProjectInfo(0, "RENDER_CHANNELS", 1, true)
  reaper.GetSetProjectInfo(0, "RENDER_BOUNDSFLAG", 2, true)   -- 时间选区
  reaper.GetSetProjectInfo(0, "RENDER_TAILFLAG", 0xFF, true)
  reaper.GetSetProjectInfo(0, "RENDER_TAILMS", TAIL_MS, true)
  reaper.GetSetProjectInfo(0, "RENDER_ADDTOPROJ", 0, true)
  reaper.GetSetProjectInfo_String(0, "RENDER_PATTERN", "", true)

  local tracks = { gtr, btr }
  for i, c in ipairs(CLIPS) do
    local t0 = (i - 1) * slot
    local t1 = t0 + clip_dur[i] + 0.3
    reaper.GetSet_LoopTimeRange(true, false, t0, t1, false)
    local dir = string.format("%s/clip_%04d", DATASET_DIR, i)
    reaper.RecursiveCreateDirectory(dir, 0)

    -- RENDER_FILE=目录, RENDER_PATTERN=文件名(无扩展) -> dir/guitar.wav, dir/bass.wav
    reaper.GetSetProjectInfo_String(0, "RENDER_FILE", dir, true)

    solo_only(gtr, tracks)
    reaper.GetSetProjectInfo_String(0, "RENDER_PATTERN", "guitar", true)
    reaper.Main_OnCommand(41824, 0)

    solo_only(btr, tracks)
    reaper.GetSetProjectInfo_String(0, "RENDER_PATTERN", "bass", true)
    reaper.Main_OnCommand(41824, 0)

    write_meta(dir, i, c.mode, clip_dur[i])
    msg(string.format("clip_%04d  mode=%-9s dur=%.1fs  rendered", i, c.mode, clip_dur[i]))
  end

  solo_only(nil, tracks)
  reaper.UpdateArrange()
  reaper.ShowMessageBox(
    string.format("完成: %d 个 clip 已渲染到\n%s\n(48k/单声道)\n\n下一步:\n"
      .. "python scripts/preprocess_mask.py --config configs/mask.json",
      #CLIPS, DATASET_DIR), "采集完成", 0)
  msg("[collect_dataset] done")
end

reaper.Undo_BeginBlock()
main()
reaper.Undo_EndBlock("ESP-GT-XFORM collect dataset", -1)
