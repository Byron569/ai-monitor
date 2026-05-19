"""
融合跌倒判断逻辑
结合规则判断（项目2）和物理特征（项目1），支持双摄像头交叉验证
"""

import time
import logging
import numpy as np
from scipy.signal import savgol_filter
from plugins.fall_engine.features import compute_all_features, get_torso_inclination
from plugins.fall_engine.config import get_config as _get_fall_config

logger = logging.getLogger(__name__)


def _cfg():
    return _get_fall_config().fall


def calculate_angle(a, b, c):
    """计算三点角度（b 为顶点）"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    if np.array_equal(a, b) or np.array_equal(b, c):
        return 180.0

    vec_ba = a - b
    vec_bc = c - b
    radians = np.arctan2(vec_bc[1], vec_bc[0]) - np.arctan2(vec_ba[1], vec_ba[0])
    angle = abs(np.degrees(radians))
    if angle > 180.0:
        angle = 360 - angle
    return angle


def check_rule_based(kp_5, aspect_ratio, angle_keypoints, bbox_area=None, initial_ar=None,
                     smoothed_angle=None, torso_inclination=None):
    """
    规则判断：宽高比 AND 角度同时触发才算（减少坐姿误报）

    torso_inclination: 躯干倾斜角（度），站立~0°，跌倒>50°
    smoothed_angle: 三点角度（度），站立~180°，跌倒<130°

    返回: bool 是否触发"可能跌倒"
    """
    ar_triggered = False
    angle_triggered = False

    # 规则 1: 宽高比（小目标跳过，用变化量代替绝对值）
    if aspect_ratio is not None:
        if bbox_area is not None and bbox_area < _cfg().get('min_bbox_area', 3000):
            pass  # 小目标宽高比不可靠，跳过
        elif initial_ar is not None and initial_ar > 0:
            ar_ratio = aspect_ratio / initial_ar
            if ar_ratio < 0.35:  # 比站姿扁了 65%
                ar_triggered = True
        elif aspect_ratio < _cfg().get('horizontal_ar_threshold', 0.6):
            ar_triggered = True

    # 规则 2: 角度判断（躯干倾斜角 OR 三点角度，任一触发即可）
    # 躯干倾斜角：直接反映"人是否倒了"，站立~0°，跌倒>50°
    if torso_inclination is not None and torso_inclination > _cfg().get('torso_inclination_threshold', 65):
        angle_triggered = True
    # 三点角度：反映身体折叠，对蜷缩有独特价值
    elif smoothed_angle is not None and smoothed_angle > 0:
        if smoothed_angle < _cfg().get('angle_threshold', 130):
            angle_triggered = True
    elif angle_keypoints is not None:
        angle = calculate_angle(
            angle_keypoints['shoulder'],
            angle_keypoints['hip'],
            angle_keypoints['knee']
        )
        if angle < _cfg().get('angle_threshold', 130):
            angle_triggered = True

    # AND 逻辑：两个条件都满足才触发
    return ar_triggered and angle_triggered


def check_physical_features(features):
    """
    物理特征判断（来自 HumanFallDetection）
    使用 EMA 平滑后的 RE/GF 值，减少噪声误报

    返回: bool 是否触发"可能跌倒"
    """
    if features is None:
        return False

    triggered = False

    # 旋转能量突然增大（使用平滑值）
    re_val = features.get('re_smoothed', features.get('re', 0))
    if re_val > _cfg().get('re_threshold', 20):
        triggered = True

    # 重力因子异常（向下加速，使用平滑值）
    gf_val = features.get('gf_smoothed', features.get('gf', 0))
    if gf_val > _cfg().get('gf_threshold', 8000):
        triggered = True

    # 头部持续下降（捕捉慢速滑倒）
    if features.get('head_descent', 0) > _cfg().get('head_descent_threshold', 0.2):
        triggered = True

    return triggered


def evaluate_fall(person_data, current_time, dual_cam_fall=None):
    """
    综合判断跌倒

    person_data: dict 包含:
        - 'kp_5': 当前 5 关键点
        - 'aspect_ratio': 当前宽高比
        - 'angle_keypoints': 用于角度计算的关键点坐标 (可选)
        - 'history': 历史 5 关键点列表
        - 'fall_state': 跌倒状态 dict

    dual_cam_fall: 另一个摄像头对该人的跌倒判断结果 (可选)

    返回: dict 包含:
        - 'fall_detected': bool
        - 'confidence': float
        - 'state': 'Normal' / 'Potential Fall' / 'Fall'
    """
    fall_state = person_data['fall_state']

    # 帧率归一化：计算 dt（帧间时间），传给特征计算
    prev_time = fall_state.get('prev_frame_time')
    if prev_time is not None:
        dt = current_time - prev_time
        dt = max(0.01, min(dt, 1.0))  # 限制在 10ms~1s 之间，防止异常值
    else:
        dt = None
    fall_state['prev_frame_time'] = current_time
    person_data['dt'] = dt

    # 计算物理特征（RE/GF 已帧率归一化）
    features = compute_all_features(person_data)

    # EMA 平滑物理特征（降噪）
    if features is not None:
        raw_re = features.get('re', 0)
        raw_gf = features.get('gf', 0)
        prev_smoothed_re = fall_state.get('ema_re', raw_re)
        prev_smoothed_gf = fall_state.get('ema_gf', raw_gf)
        ema_re = _cfg().get('ema_alpha', 0.3) * raw_re + (1 - _cfg().get('ema_alpha', 0.3)) * prev_smoothed_re
        ema_gf = _cfg().get('ema_alpha', 0.3) * raw_gf + (1 - _cfg().get('ema_alpha', 0.3)) * prev_smoothed_gf
        fall_state['ema_re'] = ema_re
        fall_state['ema_gf'] = ema_gf

        # Savitzky-Golay 二次平滑（保留信号形状，比纯 EMA 更好去噪）
        re_buf = fall_state.get('re_buffer', [])
        gf_buf = fall_state.get('gf_buffer', [])
        re_buf.append(ema_re)
        gf_buf.append(ema_gf)
        if len(re_buf) > _cfg().get('sg_window', 7):
            re_buf = re_buf[-_cfg().get('sg_window', 7):]
        if len(gf_buf) > _cfg().get('sg_window', 7):
            gf_buf = gf_buf[-_cfg().get('sg_window', 7):]
        fall_state['re_buffer'] = re_buf
        fall_state['gf_buffer'] = gf_buf

        if len(re_buf) >= _cfg().get('sg_window', 7):
            features['re_smoothed'] = float(savgol_filter(re_buf, _cfg().get('sg_window', 7), _cfg().get('sg_polyorder', 2))[-1])
            features['gf_smoothed'] = float(savgol_filter(gf_buf, _cfg().get('sg_window', 7), _cfg().get('sg_polyorder', 2))[-1])
        else:
            features['re_smoothed'] = ema_re
            features['gf_smoothed'] = ema_gf

    # 记录初始站立宽高比（滑动最大值：允许弯腰后恢复，避免 initial_ar 被弯腰拉低）
    current_ar = person_data.get('aspect_ratio')
    if current_ar is not None and not fall_state.get('is_potential_fall') and not fall_state.get('fall_detected'):
        ar_buf = fall_state.get('ar_buffer', [])
        ar_buf.append(current_ar)
        if len(ar_buf) > 30:
            ar_buf = ar_buf[-30:]
        fall_state['ar_buffer'] = ar_buf
        # 用滑动最大值作为站立基线（取最近 30 帧的最大 AR）
        fall_state['initial_ar'] = max(ar_buf)
    elif 'initial_ar' not in fall_state:
        fall_state['initial_ar'] = current_ar

    # 记录初始 body_height（头到膝中点的距离）
    kp_5 = person_data.get('kp_5')
    if 'initial_body_height' not in fall_state and kp_5 is not None:
        bh = (kp_5['KL'][1] + kp_5['KR'][1]) / 2 - kp_5['H'][1]
        if bh > 10:
            fall_state['initial_body_height'] = bh
    person_data['initial_body_height'] = fall_state.get('initial_body_height')

    # 计算 bbox 面积
    bbox = person_data.get('bbox')
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if bbox else None

    # 髋部角度计算（两种来源）
    ak = person_data.get('angle_keypoints')
    kp_5 = person_data.get('kp_5')
    hip_angle = 0

    if ak is not None:
        hip_angle = calculate_angle(ak['shoulder'], ak['hip'], ak['knee'])
    elif kp_5 is not None:
        # fallback: 用 5 关键点的 B(髋)/KL(膝)/H(头) 计算
        hip_angle = calculate_angle(kp_5['B'], kp_5['KL'], kp_5['H'])

    if hip_angle > 0:
        if 'angle_history' not in fall_state:
            fall_state['angle_history'] = []
        fall_state['angle_history'].append(hip_angle)
        if len(fall_state['angle_history']) > 3:
            fall_state['angle_history'] = fall_state['angle_history'][-3:]
    hip_angle_smoothed = float(np.mean(fall_state['angle_history'])) if fall_state.get('angle_history') else 0.0

    # 躯干倾斜角（髋→肩向量与垂直轴夹角，站立~0°，跌倒>50°）
    # 监控视角下髋→肩向量过短时不可靠，跳过
    torso_inclination = None
    if kp_5 is not None:
        torso_vec_len = np.linalg.norm(np.array(kp_5['N']) - np.array(kp_5['B']))
        body_h = abs(kp_5['H'][1] - (kp_5['KL'][1] + kp_5['KR'][1]) / 2)
        if body_h > 10 and torso_vec_len / body_h > 0.1:
            torso_inclination = get_torso_inclination(kp_5)

    initial_ar = fall_state.get('initial_ar')

    if logger.isEnabledFor(logging.DEBUG):
        hd = features.get('head_descent', 0) if features else 0
        hd_info = ""
        if kp_5 is not None and hd > 0:
            head_y = kp_5['H'][1]
            kl_y = kp_5['KL'][1]
            kr_y = kp_5['KR'][1]
            body_h = (kl_y + kr_y) / 2 - head_y
            hd_info = f" head_y:{head_y:.0f} knees_avg:{(kl_y+kr_y)/2:.0f} body_h:{body_h:.0f}"
        re_raw = features.get('re', 0) if features else 0
        gf_raw = features.get('gf', 0) if features else 0
        re_s = features.get('re_smoothed', re_raw) if features else 0
        gf_s = features.get('gf_smoothed', gf_raw) if features else 0
        logger.debug(f"PID:{person_data.get('pid')} dt:{dt or 0:.3f}s "
              f"AR:{person_data.get('aspect_ratio', 0):.2f} init_AR:{initial_ar or 0:.2f} "
              f"angle:{hip_angle:.1f}(s:{hip_angle_smoothed:.1f}) src:{'ak' if ak else 'kp5' if kp_5 else 'none'} "
              f"HD:{hd:.2f}{hd_info} "
              f"RE:{re_raw:.1f}→{re_s:.1f} GF:{gf_raw:.0f}→{gf_s:.0f} "
              f"hist:{len(fall_state.get('trigger_history', []))} "
              f"guard:{'BLOCK' if initial_ar is not None and initial_ar < _cfg().get('min_standing_ar', 1.2) else 'OK'}")

    # 保护：初始状态不是站立（AR < 1.2）→ 跳过 AR 变化检测，但允许物理特征检测
    initial_ar_low = initial_ar is not None and initial_ar < _cfg().get('min_standing_ar', 1.2)

    # 三路检测（任一路径触发即为"可能跌倒"）
    current_ar = person_data.get('aspect_ratio')

    # 路径 1: 几何检测（AR + 角度同时变化）→ 前倒/后倒/侧倒
    # 初始 AR 太低（非站立）时跳过，因为 AR 变化不可靠
    if initial_ar_low:
        rule_triggered = False
    else:
        rule_triggered = check_rule_based(
            person_data.get('kp_5'), current_ar, ak,
            bbox_area=bbox_area, initial_ar=initial_ar,
            smoothed_angle=hip_angle_smoothed,
            torso_inclination=torso_inclination
        )

    # 路径 2: 物理检测（RE 或 GF 强信号）→ 快速摔倒/蜷缩倒
    physics_triggered = check_physical_features(features)

    # 计算 ar_ratio（供路径 2 守卫和路径 3 使用）
    ar_ratio = None
    if initial_ar is not None and initial_ar > 0 and current_ar is not None:
        ar_ratio = current_ar / initial_ar

    # 路径 3: AR 剧变 + 头部下降 → 侧倒（角度可能不够低）
    ar_dramatic = False
    if not initial_ar_low and ar_ratio is not None:
        if ar_ratio < 0.35:
            ar_dramatic = True
    # 初始 AR 低时：只要当前 AR 很低且有头部下降，也算触发
    elif initial_ar_low and current_ar is not None and current_ar < _cfg().get('horizontal_ar_threshold', 0.6):
        ar_dramatic = True
    head_drop_triggered = features and features.get('head_descent', 0) > _cfg().get('head_descent_threshold', 0.2)
    side_fall_triggered = ar_dramatic and head_drop_triggered

    # 路径 4: 已经在地上（AR 极低且持续多帧，无需运动特征）
    # 用独立帧计数器，不依赖 trigger_history（避免鸡生蛋问题）
    # AR < 0.4 排除弯腰/半蹲（它们 AR 通常 0.5~0.8）
    already_down = False
    if initial_ar_low and current_ar is not None and current_ar < 0.4:
        fall_state['ar_low_frames'] = fall_state.get('ar_low_frames', 0) + 1
        if fall_state['ar_low_frames'] >= 10 and not fall_state.get('fall_detected', False):
            already_down = True
    else:
        fall_state['ar_low_frames'] = 0

    # 物理路径需配合几何信号：走路/弯腰时 RE/GF 超阈值但 AR 没变，不应触发
    # 头部下降除外（捕捉慢速滑倒，其 AR 变化可能不大）
    ar_changed = ar_ratio is not None and ar_ratio < 0.9
    physics_with_geometry = physics_triggered and (ar_changed or head_drop_triggered)

    # 任一路径触发则标记为"可能跌倒"
    is_potential_fall = bool(rule_triggered or physics_with_geometry or side_fall_triggered or already_down)

    # 路径诊断（只在有触发且 debug 级别时打印）
    if logger.isEnabledFor(logging.DEBUG) and (is_potential_fall or fall_state.get('is_potential_fall')):
        ar_ratio_val = current_ar / initial_ar if initial_ar and current_ar else 0
        logger.debug(f"  [PATH] P1:{rule_triggered} P2:{physics_triggered}(geo:{physics_with_geometry}) P3:{side_fall_triggered} P4:{already_down} "
              f"ar_ratio:{ar_ratio_val:.2f} guard:{'LOW' if initial_ar_low else 'OK'} "
              f"win:{sum(fall_state.get('trigger_history',[]))}/{len(fall_state.get('trigger_history',[]))} "
              f"pot:{fall_state.get('is_potential_fall')} det:{fall_state.get('fall_detected')}")

    # 滑动时间窗口：记录最近 N 帧的触发状态
    if 'trigger_history' not in fall_state:
        fall_state['trigger_history'] = []

    fall_state['trigger_history'].append(is_potential_fall)
    if len(fall_state['trigger_history']) > _cfg().get('window_size', 10):
        fall_state['trigger_history'] = fall_state['trigger_history'][-_cfg().get('window_size', 10):]

    # 连续触发帧计数（允许 1 帧间隙，防止倒地过程中检测框抖动导致计数归零）
    if is_potential_fall:
        fall_state['consecutive_triggers'] = fall_state.get('consecutive_triggers', 0) + 1
        fall_state['trigger_gap_count'] = 0
    else:
        gap = fall_state.get('trigger_gap_count', 0) + 1
        fall_state['trigger_gap_count'] = gap
        if gap >= 2:
            fall_state['consecutive_triggers'] = 0

    trigger_ratio = sum(fall_state['trigger_history']) / len(fall_state['trigger_history'])
    consecutive_ok = fall_state.get('consecutive_triggers', 0) >= _cfg().get('min_consecutive_triggers', 3)
    window_triggered = trigger_ratio >= _cfg().get('window_trigger_ratio', 0.5) and consecutive_ok

    if window_triggered:
        if not fall_state['is_potential_fall']:
            fall_state['is_potential_fall'] = True
            fall_state['fall_start_time'] = current_time
        # 记录头部最低位置（y 越大越低）
        curr_kp = person_data.get('kp_5')
        if curr_kp is not None:
            head_y = curr_kp['H'][1]
            if 'min_head_y' not in fall_state or head_y > fall_state['min_head_y']:
                fall_state['min_head_y'] = head_y
    else:
        # 已确认跌倒后，滑动窗口失效不应重置状态（人躺在地上时物理信号会消失）
        if not fall_state.get('fall_detected', False):
            fst = fall_state.get('fall_start_time')
            if fst is not None and current_time - fst < _cfg().get('min_fall_pose_duration', 1.0) - 0.01:
                # fall_start_time 已设置但持续时间未到 → 保持状态，给人倒下后静止的场景留时间
                pass
            elif fall_state['is_potential_fall']:
                fall_state['is_potential_fall'] = False
                fall_state['trigger_history'] = []
                fall_state['consecutive_triggers'] = 0
                fall_state['trigger_gap_count'] = 0
                fall_state.pop('min_head_y', None)
                # 注意：不清理 fall_start_time，留给下面的持续时间确认检查

    # 持续时间确认 + 头部回弹检测（即使 is_potential_fall 已被重置，只要有 fall_start_time 就检查）
    fall_detected = fall_state.get('fall_detected', False)  # 保持之前的跌倒状态
    fst_for_check = fall_state.get('fall_start_time')
    # 恢复守卫：AR 恢复到站立水平 且 窗口未触发 → 说明人站起来了，清除 fall_start_time
    if (fst_for_check is not None and not window_triggered
            and current_ar is not None and initial_ar is not None and current_ar > initial_ar * 0.7):
        fall_state['fall_start_time'] = None
        fst_for_check = None
    if fst_for_check is not None:
        duration = current_time - fall_state['fall_start_time']
        if duration >= _cfg().get('min_fall_pose_duration', 1.0) - 0.01:  # 浮点精度容差
            # 检查头部是否已回弹（弯腰后站起 vs 真正摔倒）
            # 需要连续多帧回弹才判定，防止单帧噪声取消跌倒
            curr_kp = person_data.get('kp_5')
            rebound = False
            if curr_kp is not None and 'min_head_y' in fall_state:
                head_drop = fall_state['min_head_y'] - curr_kp['H'][1]
                body_h = abs(curr_kp['H'][1] - (curr_kp['KL'][1] + curr_kp['KR'][1]) / 2)
                if body_h > 10 and head_drop / body_h > _cfg().get('rebound_threshold', 0.3):
                    rebound = True
            # 多帧回弹确认
            if rebound:
                fall_state['rebound_count'] = fall_state.get('rebound_count', 0) + 1
            else:
                fall_state['rebound_count'] = 0
            if fall_state.get('rebound_count', 0) < _cfg().get('rebound_frames', 2):
                fall_detected = True
            else:
                # 回弹确认 → 不是真正跌倒，清理 fall_start_time
                fall_state['fall_start_time'] = None

    # 高置信度快速通道：四个条件同时强力触发时，跳过持续时间要求
    if not fall_detected and features is not None:
        ar = person_data.get('aspect_ratio')
        fast_angle = hip_angle_smoothed > 0 and hip_angle_smoothed < _cfg().get('fast_angle_threshold', 120)
        re_val = features.get('re_smoothed', features.get('re', 0))
        gf_val = features.get('gf_smoothed', features.get('gf', 0))

        if (ar is not None and ar < _cfg().get('horizontal_ar_threshold', 0.6) and
            fast_angle and
            re_val > _cfg().get('fast_re_threshold', 20) and
            gf_val > _cfg().get('fast_gf_threshold', 15000)):
            fall_detected = True

    # 恢复检测：如果已确认跌倒但人站起来了，重置状态
    # 需要连续多帧 AR 恢复才重置，防止单帧噪声误重置
    if fall_detected and initial_ar is not None and initial_ar > _cfg().get('min_standing_ar', 1.2):
        current_ar = person_data.get('aspect_ratio')
        if current_ar is not None and current_ar > initial_ar * 0.7:
            fall_state['recovery_count'] = fall_state.get('recovery_count', 0) + 1
        else:
            fall_state['recovery_count'] = 0
        if fall_state.get('recovery_count', 0) >= _cfg().get('recovery_frames', 3):
            fall_detected = False
            fall_state['is_potential_fall'] = False
            fall_state['fall_start_time'] = None
            fall_state['trigger_history'] = []
            fall_state['consecutive_triggers'] = 0
            fall_state['trigger_gap_count'] = 0
            fall_state.pop('min_head_y', None)
            fall_state['recovery_count'] = 0

    fall_state['fall_detected'] = fall_detected

    # 确定状态和置信度
    if fall_detected:
        if _cfg().get('dual_cam_confirm', True) and dual_cam_fall is not None:
            # 双摄像头交叉验证
            if dual_cam_fall:
                confidence = _cfg().get('dual_cam_fall_confidence', 0.95)
                state = "FALL (Confirmed)"
            else:
                confidence = _cfg().get('single_cam_fall_confidence', 0.6)
                state = "FALL (Single Cam)"
        else:
            confidence = _cfg().get('single_cam_fall_confidence', 0.6)
            state = "FALL"
    elif fall_state['is_potential_fall']:
        confidence = 0.3
        state = "Potential Fall"
    else:
        confidence = 0.0
        state = "Normal"

    return {
        'fall_detected': fall_detected,
        'confidence': confidence,
        'state': state,
    }
