# 배울 개념: enter(20°)와 exit(12°)를 다르게 두는 히스테리시스 때문에, 경계값 근처에서 알림이 깜빡이지 않습니다.
# 이동 중에는 시간을 "멈추기만" 하고(활동 게이트), 3초 넘게 이동하면 후보를 초기화.

NORMAL, CANDIDATE, CONFIRMED, RECOVERY = "정상", "후보", "확정", "회복확인"

class RuleStateMachine:
    def __init__(self, enter, exit, hold_s=5.0, recovery_s=3.0, gate_activity=("static",), gate_reset_s=3.0):
        self.enter, self.exit = enter, exit
        self.hold_s, self.recovery_s = hold_s, recovery_s
        self.gate_activity, self.gate_reset_s = gate_activity, gate_reset_s
        self.state = NORMAL
        self.elapsed = 0.0 # 후보 상태 누적 시간
        self.gate_out = 0.0 # 게이트 밖 누적 시간
        self.recover = 0.0 # 회복확인 누적 시간

    def step(self, value, activity, dt):
        """한 스텝 갱신. 알림을 보내야 하면 True 반환"""
        gated = activity in self.gate_activity

        if self.state == NORMAL:
            if gated and value >= self.enter:
                self.state, self.elapsed, self.gate_out = CANDIDATE, 0.0, 0.0

        elif self.state == CANDIDATE:
            if not gated:                          # 이동 중: 시간 정지
                self.gate_out += dt
                if self.gate_out >= self.gate_reset_s:
                    self.state = NORMAL
                return False
            self.gate_out = 0.0
            if value <= self.exit:
                self.state = NORMAL
            else:
                self.elapsed += dt
                if self.elapsed >= self.hold_s:
                    self.state = CONFIRMED
                    return True                    # ← 알림 1회

        elif self.state == CONFIRMED:
            if value <= self.exit:
                self.state, self.recover = RECOVERY, 0.0

        elif self.state == RECOVERY:
            if value > self.exit:
                self.state = CONFIRMED             # 다시 굽힘: 알림 없이 복귀
            else:
                self.recover += dt
                if self.recover >= self.recovery_s:
                    self.state = NORMAL
        return False


# ======================================================================
# 변수·함수 설명
# ======================================================================
#
# [상태 4가지]  (NORMAL → CANDIDATE → CONFIRMED → RECOVERY → NORMAL 순환)
#   NORMAL     정상. 감시 대상 자세가 아니다.
#   CANDIDATE  후보. 기준(enter) 이상으로 굽혀서 "정말 오래 유지되는지" 시간을 재는 중.
#   CONFIRMED  확정. hold_s초 이상 유지돼 알림을 이미 1번 보낸 상태.
#              (같은 자세가 이어져도 알림을 반복하지 않는다)
#   RECOVERY   회복확인. 자세가 풀렸지만, 잠깐 풀린 건지 완전히 풀린 건지 확인 중.
#
# [생성자 인자]
#   enter          후보로 진입하는 각도(°) 문턱. 예: 20
#   exit           후보/확정에서 벗어나는 각도(°) 문턱. 예: 12
#                  enter > exit 로 두는 것이 히스테리시스다. 각도가 15° 근처에서
#                  흔들려도 enter(20)를 넘어야 시작하고 exit(12) 밑으로 내려가야
#                  끝나므로, 경계에서 상태가 깜빡거리지 않는다.
#   hold_s         후보 상태를 이 시간(초) 누적해야 알림 확정. 기본 5초.
#   recovery_s     회복확인에서 exit 이하를 이 시간(초) 유지해야 정상 복귀. 기본 3초.
#   gate_activity  판정을 허용하는 활동 상태 목록. 기본 ("static",) = 정지일 때만.
#                  걷기 등 이동 중 순간 가속으로 인한 오판을 막는 "활동 게이트".
#   gate_reset_s   후보 중 게이트 밖(이동)이 이 시간(초) 이상 이어지면 후보를 취소. 기본 3초.
#
# [내부 변수]
#   state     현재 상태 (위 4가지 중 하나)
#   elapsed   후보 상태에서 쌓인 시간(초). hold_s에 도달하면 확정.
#   gate_out  후보 중 이동 상태로 보낸 연속 시간(초). gate_reset_s에 도달하면 초기화.
#   recover   회복확인 상태에서 쌓인 시간(초). recovery_s에 도달하면 정상 복귀.
#
# [step(value, activity, dt)]  한 샘플(또는 한 스트라이드)마다 호출
#   value     감시할 값, 보통 구간 굴곡각(°). enter/exit와 같은 단위.
#   activity  현재 활동 분류 ("static" / "moving" / "transition")
#   dt        이전 호출 이후 경과 시간(초). 시간 누적에 사용.
#   gated     activity가 gate_activity에 들어 있으면 True (= 판정 가능한 상태)
#   반환      True = 이번 호출에서 알림을 1회 보내야 함, False = 보내지 않음.
#             알림은 CANDIDATE → CONFIRMED로 넘어가는 순간에만 True.
#
# [상태별 동작]
#   NORMAL     게이트 안(정지)이고 value >= enter 이면 CANDIDATE로.
#              이때 elapsed, gate_out을 0으로 리셋한다 (이전 후보의 찌꺼기 방지).
#   CANDIDATE  이동 중이면 시간(elapsed)을 멈추고 gate_out만 쌓는다.
#              gate_out이 gate_reset_s를 넘으면 NORMAL로 취소.
#              정지로 돌아오면 gate_out=0. value <= exit면 NORMAL로 취소,
#              아니면 elapsed += dt, hold_s 도달 시 CONFIRMED + 알림.
#   CONFIRMED  value <= exit가 되면 RECOVERY로.
#   RECOVERY   value > exit(다시 굽힘)면 알림 없이 CONFIRMED로 복귀.
#              아니면 recover += dt, recovery_s 도달 시 NORMAL.
