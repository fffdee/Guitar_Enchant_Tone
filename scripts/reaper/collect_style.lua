--[[
  ESP-GT-XFORM 演奏法导向数据采集 v2 (Reaper ReaScript / Lua)
  ------------------------------------------------------------------
  思路：你在【目标乐器轨】上按该乐器的真实演奏法准备 MIDI
        (bass: baseline / slap 等；ukulele: 扫弦 / 分解和弦 等)，
        本脚本：读取目标轨 MIDI → 按"大间隔(静音)"自动切成多个 clip →
        把每个 clip 的音符【镜像】到 guitar 源轨(同音高) → 逐轨 solo 渲染
        严格对齐的 guitar.wav 与 <target>.wav。

  关键约束(频域谱掩码不能移调)：源吉他与目标必须弹【相同音高】。
    因此目标 MIDI 请尽量写在【吉他可发声音域 E2..E6 = MIDI 40..88】内；
    需要更低的 bass / 更高的 ukulele 音区，请在【推理时用"移频(半音)"滑杆】调，
    而不是在训练数据里移调(否则掩码学不出来)。脚本会统计并提示越界音符数。

  "每次大间隔就是一次数据"：相邻音符间静音 > GAP_SPLIT 秒即切一个新 clip。

  输出(新数据集, 旧 dataset/raw 不动也不用)：
    dataset/raw_style/clip_XXXX/{guitar.wav, <target>.wav, meta.json}  (48k/单声道)
    多次运行自动接着已有最大编号追加(可分乐器、分批采集)。

  轨道命名约定(脚本按名字找轨；找不到则新建并挂默认音源)：
    "guitar"(源) / "bass" / "ukulele"。只处理【有 MIDI】的目标轨。

  运行前(只需一次)：
    - File > Render：Format=WAV、关 Normalize、勾 Tail，关掉对话框(REAPER 记住)。
    - 各 Ample 音源：关 Amp/Cab/效果(干 DI)、关 Humanize/随机时间；确认音色库已加载。
  运行：Actions > Show action list > Load... 选本文件 > Run。
]]--

-------------------- 配置 --------------------
local DATASET_DIR = "e:/DeepLearn/ESP_XFORM/dataset/raw_style"
local GUITAR_NAME = "guitar"
local GUITAR_FX   = "Ample Guitar M"
-- 目标乐器：{ 轨名, 音源名(TrackFX_AddByName 子串匹配) }
local TARGETS = {
  { name = "bass",    fx = "Ample Bass P" },
  { name = "ukulele", fx = "Ample Ethno U" },
}
local GAP_SPLIT  = 1.5     -- 相邻音符静音 > 该秒数 => 切新 clip ("大间隔")
local MIN_CLIP   = 1.0     -- 小于该时长的片段丢弃(秒)
local PRE_ROLL   = 0.10    -- clip 起点前留白(秒)，避免切掉起音
local TAIL_S     = 0.40    -- clip 末尾保留尾音(秒)
local LO_FIT, HI_FIT = 40, 88   -- 吉他可发声音域(仅用于越界统计提示)
local SR         = 48000
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
    if nm == name then return tr, i end
  end
  return nil, -1
end

local function ensure_track(name)
  local tr = find_track_by_name(name)
  if not tr then
    local idx = reaper.CountTracks(0)
    reaper.InsertTrackAtIndex(idx, true)
    tr = reaper.GetTrack(0, idx)
    reaper.GetSetMediaTrackInfo_String(tr, "P_NAME", name, true)
  end
  return tr
end

local function ensure_instrument(track, fxname)
  local idx = reaper.TrackFX_GetInstrument(track)
  if idx < 0 then idx = reaper.TrackFX_AddByName(track, fxname, false, -1) end
  if idx >= 0 then reaper.TrackFX_SetEnabled(track, idx, true) end
  return idx
end

local function clear_track_items(track)
  for i = reaper.CountTrackMediaItems(track) - 1, 0, -1 do
    reaper.DeleteTrackMediaItem(track, reaper.GetTrackMediaItem(track, i))
  end
end

-- 读取一条轨上所有 item / take 的全部音符 (绝对时间, 秒)
local function read_track_notes(track)
  local notes = {}
  for it = 0, reaper.CountTrackMediaItems(track) - 1 do
    local item = reaper.GetTrackMediaItem(track, it)
    local take = reaper.GetActiveTake(item)
    if take and reaper.TakeIsMIDI(take) then
      local _, ncnt = reaper.MIDI_CountEvts(take)
      for i = 0, ncnt - 1 do
        local ok, _, _, sppq, eppq, chan, pitch, vel = reaper.MIDI_GetNote(take, i)
        if ok then
          notes[#notes + 1] = {
            s = reaper.MIDI_GetProjTimeFromPPQPos(take, sppq),
            e = reaper.MIDI_GetProjTimeFromPPQPos(take, eppq),
            chan = chan, pitch = pitch, vel = vel,
          }
        end
      end
    end
  end
  table.sort(notes, function(a, b) return a.s < b.s end)
  return notes
end

-- 按大间隔切 clip；返回 { {notes=.., t0=.., t1=..}, ... }
local function split_clips(notes, gap, min_clip)
  local clips, cur, run_end = {}, nil, nil
  local function flush()
    if cur and #cur > 0 then
      local t0, t1 = cur[1].s, cur[1].e
      for _, nt in ipairs(cur) do if nt.e > t1 then t1 = nt.e end end
      if (t1 - t0) >= min_clip then clips[#clips + 1] = { notes = cur, t0 = t0, t1 = t1 } end
    end
  end
  for _, nt in ipairs(notes) do
    if cur == nil then
      cur, run_end = { nt }, nt.e
    elseif nt.s - run_end > gap then
      flush(); cur, run_end = { nt }, nt.e
    else
      cur[#cur + 1] = nt
      if nt.e > run_end then run_end = nt.e end
    end
  end
  flush()
  return clips
end

local function write_notes(take, notes)
  for _, nt in ipairs(notes) do
    local p0 = reaper.MIDI_GetPPQPosFromProjTime(take, nt.s)
    local p1 = reaper.MIDI_GetPPQPosFromProjTime(take, nt.e)
    reaper.MIDI_InsertNote(take, false, false, p0, p1, nt.chan or 0, nt.pitch, nt.vel, true)
  end
  reaper.MIDI_Sort(take)
end

local function solo_only(target, tracks)
  for _, tr in ipairs(tracks) do
    reaper.SetMediaTrackInfo_Value(tr, "I_SOLO", tr == target and 1 or 0)
  end
end

-- 扫描数据集已有 clip_#### 的最大编号 (跨多次运行追加)
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

local function write_meta(dir, idx, target_name, target_fx, dur, n_notes, oob)
  local f = io.open(dir .. "/meta.json", "w")
  if not f then return end
  f:write(string.format(
    '{\n  "sample_rate": %d,\n  "channels": 1,\n  "instruments": ["guitar", "%s"],\n'
    .. '  "source": "guitar",\n  "targets": ["%s"],\n  "clip": %d,\n  "mode": "style",\n'
    .. '  "duration_s": %.3f,\n  "n_notes": %d,\n  "out_of_guitar_range": %d,\n'
    .. '  "guitar_fx": "%s",\n  "%s_fx": "%s"\n}\n',
    SR, target_name, target_name, idx, dur, n_notes, oob, GUITAR_FX, target_name, target_fx))
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

  local gtr = ensure_track(GUITAR_NAME)
  if ensure_instrument(gtr, GUITAR_FX) < 0 then
    reaper.ShowMessageBox("吉他音源缺失: " .. GUITAR_FX, "错误", 0); return
  end

  -- 收集所有相关轨(用于 solo 互斥)
  local all_tracks = { gtr }
  local active_targets = {}
  for _, tg in ipairs(TARGETS) do
    local ttr = find_track_by_name(tg.name)
    if ttr then
      if ensure_instrument(ttr, tg.fx) < 0 then
        msg(string.format("跳过 %s：音源 %s 加载失败", tg.name, tg.fx))
      else
        local notes = read_track_notes(ttr)
        if #notes > 0 then
          active_targets[#active_targets + 1] = { tg = tg, tr = ttr, notes = notes }
          all_tracks[#all_tracks + 1] = ttr
        else
          msg(string.format("跳过 %s：该轨无 MIDI 音符", tg.name))
        end
      end
    end
  end

  if #active_targets == 0 then
    reaper.ShowMessageBox("没有发现带 MIDI 的目标轨(bass / ukulele)。\n请在目标乐器轨上准备好 MIDI 后重试。",
      "无可采集数据", 0)
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
  local total_clips = 0

  for _, at in ipairs(active_targets) do
    local tg, ttr, notes = at.tg, at.tr, at.notes
    local clips = split_clips(notes, GAP_SPLIT, MIN_CLIP)

    -- 越界统计(吉他无法发声的音符)
    local oob_total = 0
    local last_end = 0.0
    for _, nt in ipairs(notes) do
      if nt.pitch < LO_FIT or nt.pitch > HI_FIT then oob_total = oob_total + 1 end
      if nt.e > last_end then last_end = nt.e end
    end
    if oob_total > 0 then
      msg(string.format("[警告] %s: %d/%d 个音符超出吉他音域[%d..%d]，这些音吉他将无声(配对会偏弱)。"
        .. "建议把目标 MIDI 写进该音域，推理时再用移频滑杆调音区。",
        tg.name, oob_total, #notes, LO_FIT, HI_FIT))
    end

    -- 把目标音符【原样镜像】到吉他源轨(同音高)
    clear_track_items(gtr)
    local gitem = reaper.CreateNewMIDIItemInProj(gtr, 0, last_end + 1.0, false)
    local gtake = reaper.GetActiveTake(gitem)
    write_notes(gtake, notes)

    msg(string.format("== 目标 %s：%d 个音符 -> 切出 %d 个 clip ==", tg.name, #notes, #clips))

    for _, c in ipairs(clips) do
      local t0 = math.max(0.0, c.t0 - PRE_ROLL)
      local t1 = c.t1 + 0.05
      reaper.GetSet_LoopTimeRange(true, false, t0, t1, false)
      local dir = string.format("%s/clip_%04d", DATASET_DIR, clip_idx)
      reaper.RecursiveCreateDirectory(dir, 0)
      reaper.GetSetProjectInfo_String(0, "RENDER_FILE", dir, true)

      solo_only(gtr, all_tracks)
      reaper.GetSetProjectInfo_String(0, "RENDER_PATTERN", "guitar", true)
      reaper.Main_OnCommand(41824, 0)

      solo_only(ttr, all_tracks)
      reaper.GetSetProjectInfo_String(0, "RENDER_PATTERN", tg.name, true)
      reaper.Main_OnCommand(41824, 0)

      write_meta(dir, clip_idx, tg.name, tg.fx, c.t1 - c.t0, #c.notes, oob_total)
      msg(string.format("  clip_%04d  %-8s  %.2f..%.2fs (%.1fs, %d 音)",
        clip_idx, tg.name, c.t0, c.t1, c.t1 - c.t0, #c.notes))
      clip_idx = clip_idx + 1
      total_clips = total_clips + 1
    end

    clear_track_items(gtr)
  end

  solo_only(nil, all_tracks)
  reaper.UpdateArrange()
  reaper.ShowMessageBox(
    string.format("完成：本次新增 %d 个 clip 到\n%s\n(48k/单声道, 按大间隔=%.1fs 切分)\n\n"
      .. "下一步(回到 Cursor)：\npython scripts/preprocess_mask.py --config configs/mask_style.json "
      .. "--instruments configs/instruments.json",
      total_clips, DATASET_DIR, GAP_SPLIT),
    "演奏法数据采集完成", 0)
  msg("[collect_style] done, +" .. total_clips .. " clips")
end

reaper.Undo_BeginBlock()
main()
reaper.Undo_EndBlock("ESP-GT-XFORM collect style dataset", -1)
