// AnzZheng 로거: ESP32 + MPU-6050 2개 → 50 Hz CSV를 USB 시리얼로 보낸다.
// 노트북에서 받기: python -m ml.app.record --port /dev/cu.usbserial-XXXX
//
// [배선]  두 센서를 같은 I2C 선에 병렬로 연결한다 (납땜 없이 점퍼선만)
//   ESP32 3V3 → 두 센서 VCC        ESP32 GND → 두 센서 GND
//   ESP32 G21 → 두 센서 SDA        ESP32 G22 → 두 센서 SCL
//   상부(T12/L1) 센서 AD0 → GND   (주소 0x68)
//   하부(S1)     센서 AD0 → 3V3   (주소 0x69)   ← 이것 때문에 두 개가 구분된다
//
// [부착]  부품면이 바깥(등 뒤)을 보게, 보드의 X 화살표가 머리 쪽을 향하게 붙인다.
//   그러면 센서 x = 척추 위쪽, z = 등 바깥쪽 → ml/ 코드의 축 규약과 같다.
//
// [출력]  한 줄 = 한 샘플:  seq,t_us,u_ax,u_ay,u_az,u_gx,u_gy,u_gz,l_ax,l_ay,l_az,l_gx,l_gy,l_gz
//   값은 원시 정수(LSB). 가속도 ±4g → 8192 LSB/g, 자이로 ±500°/s → 65.5 LSB/(°/s)
//   (ml/signal/process.py의 ACC_LSB, GYR_LSB와 같아야 한다)
//   '#'으로 시작하는 줄은 상태 메시지. 노트북 쪽은 화면에만 띄우고 저장하지 않는다.
#include <Wire.h>
#include "esp_timer.h"

const uint8_t ADDR_UPPER = 0x68;
const uint8_t ADDR_LOWER = 0x69;
const uint32_t PERIOD_US = 20000;   // 50 Hz
const uint32_t I2C_HZ = 100000;     // 선이 30 cm쯤 되면 400 kHz보다 100 kHz가 안정적

uint32_t seq = 0;
int64_t next_t = 0;

bool writeReg(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

int readReg(uint8_t addr, uint8_t reg) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return -1;
  if (Wire.requestFrom(addr, (uint8_t)1) != 1) return -1;
  return Wire.read();
}

bool setupMpu(uint8_t addr, const char* name) {
  int who = readReg(addr, 0x75);    // WHO_AM_I. 정품 0x68, 호환 칩은 0x70·0x72·0x98일 수 있다
  if (who < 0) {
    Serial.printf("# ERROR %s(0x%02X) 응답 없음: 배선·AD0 확인\n", name, addr);
    return false;
  }
  Serial.printf("# %s(0x%02X) WHO_AM_I=0x%02X\n", name, addr, who);
  bool ok = writeReg(addr, 0x6B, 0x01)    // PWR_MGMT_1: 잠 깨우기 + 자이로 클록
         && writeReg(addr, 0x19, 0x00)    // SMPLRT_DIV: 내부 1 kHz (우리는 50 Hz로 읽기만)
         && writeReg(addr, 0x1A, 0x04)    // CONFIG: 저역통과 약 20 Hz (50 Hz 샘플링 앨리어싱 방지)
         && writeReg(addr, 0x1B, 0x08)    // GYRO_CONFIG: ±500 °/s
         && writeReg(addr, 0x1C, 0x08);   // ACCEL_CONFIG: ±4 g
  return ok;
}

// 0x3B부터 14바이트: ax ay az temp gx gy gz (각 16비트, 상위 바이트 먼저)
bool readMpu(uint8_t addr, int16_t out[6]) {
  Wire.beginTransmission(addr);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(addr, (uint8_t)14) != 14) return false;
  int16_t v[7];
  for (int i = 0; i < 7; i++) v[i] = (int16_t)((Wire.read() << 8) | Wire.read());
  out[0] = v[0]; out[1] = v[1]; out[2] = v[2];   // 가속도
  out[3] = v[4]; out[4] = v[5]; out[5] = v[6];   // 자이로 (v[3]은 온도라 건너뜀)
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Wire.begin(21, 22, I2C_HZ);
  Serial.println("# AnzZheng logger v1  fs=50Hz acc=+-4g gyr=+-500dps");
  bool ok = setupMpu(ADDR_UPPER, "upper") & setupMpu(ADDR_LOWER, "lower");
  if (!ok) Serial.println("# ERROR 센서 초기화 실패. 고친 뒤 EN(리셋) 버튼을 누르세요");
  Serial.println("# seq,t_us,u_ax,u_ay,u_az,u_gx,u_gy,u_gz,l_ax,l_ay,l_az,l_gx,l_gy,l_gz");
  next_t = esp_timer_get_time();
}

void loop() {
  // 고정 주기: delay(20)은 읽기·출력 시간만큼 밀려서 실제로는 50 Hz보다 느려진다
  next_t += PERIOD_US;
  while (esp_timer_get_time() < next_t) {}
  int64_t t = esp_timer_get_time();   // 부팅 후 µs (64비트라 넘침 걱정 없음)

  int16_t u[6], l[6];
  bool ok = readMpu(ADDR_UPPER, u) && readMpu(ADDR_LOWER, l);
  if (!ok) {
    Serial.printf("# i2c_err seq=%lu\n", (unsigned long)seq);   // 이 샘플은 빠진다 (노트북이 seq로 감지)
  } else {
    Serial.printf("%lu,%lld,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                  (unsigned long)seq, (long long)t,
                  u[0], u[1], u[2], u[3], u[4], u[5], l[0], l[1], l[2], l[3], l[4], l[5]);
  }
  seq++;
}
