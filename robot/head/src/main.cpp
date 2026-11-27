// The Machine's head. It listens for one-line commands over WiFi (TCP) and USB serial:
//   P90 T60   look at pan 90, tilt 60 (either part may be left out)
//   S300      turn at 300 degrees per second
//   ?         report where it is looking
// and answers every command with a line like "OK P90 T60".

#include <Arduino.h>
#include <ArduinoOTA.h>
#include <ESP32Servo.h>
#include <ESPmDNS.h>
#include <WiFi.h>

#include "config.h"

Servo pan, tilt;
WiFiServer server(PORT);
WiFiClient client;
String fromSerial, fromClient;
bool moving = false;
unsigned long lastLog = 0;

float panNow = 90, tiltNow = 90, panGoal = 90, tiltGoal = 90, speed = SPEED;
unsigned long lastStep = 0, lastWifiTry = 0;
bool online = false;

void report(Print &out) {
  out.printf("OK P%.0f T%.0f\r\n", panGoal, tiltGoal);
}

void handle(String cmd, Print &out, const char *from) {
  cmd.trim();
  cmd.toUpperCase();
  if (cmd.isEmpty()) return;
  int i = 0;
  while (i < (int)cmd.length()) {
    char c = cmd[i++];
    int end = i;
    while (end < (int)cmd.length() && cmd[end] != ' ') end++;
    float v = cmd.substring(i, end).toFloat();
    if (c == 'P') panGoal = constrain(v, PAN_MIN, PAN_MAX);
    if (c == 'T') tiltGoal = constrain(v, TILT_MIN, TILT_MAX);
    if (c == 'S') speed = constrain(v, 10, 1000);
    i = end + 1;
  }
  if (millis() - lastLog > 250) {                        // at most four lines a second
    lastLog = millis();
    Serial.printf("<- %-12s from %s  goal P%.0f T%.0f\r\n", cmd.c_str(), from, panGoal, tiltGoal);
  }
  report(out);
}

void readLines(Stream &in, String &buf, Print &out, const char *from) {
  while (in.available()) {
    char c = in.read();
    if (c == '\n' || c == '\r') {
      handle(buf, out, from);
      buf = "";
    } else if (buf.length() < 64) {
      buf += c;
    }
  }
}

void connectWifi() {
#ifdef WIFI_SSID
  if (WiFi.status() == WL_CONNECTED || millis() - lastWifiTry < 10000) return;
  lastWifiTry = millis();
  Serial.printf("wifi: joining %s\r\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.setHostname(HOSTNAME);
  WiFi.begin(WIFI_SSID, WIFI_PASS, WIFI_CHANNEL);
#endif
}

void comeOnline() {
  online = true;
  Serial.printf("wifi: %s, listening on port %d\r\n", WiFi.localIP().toString().c_str(), PORT);
  MDNS.begin(HOSTNAME);
  MDNS.addService("machine-head", "tcp", PORT);
  ArduinoOTA.setHostname(HOSTNAME);
  ArduinoOTA.begin();                                // update the firmware over WiFi
  server.begin();
  server.setNoDelay(true);
}

// Move both servos toward their goals, no faster than `speed`.
void step() {
  unsigned long now = millis();
  float dt = (now - lastStep) / 1000.0;
  if (dt < 0.01) return;
  lastStep = now;
  float most = speed * dt;
  panNow += constrain(panGoal - panNow, -most, most);
  tiltNow += constrain(tiltGoal - tiltNow, -most, most);
  pan.write(round(panNow));
  tilt.write(round(tiltNow));
  bool still = fabs(panGoal - panNow) < 0.5 && fabs(tiltGoal - tiltNow) < 0.5;
  if (moving && still) Serial.printf("   at P%.0f T%.0f\r\n", panNow, tiltNow);
  moving = !still;
}

// Turn left and right, up and down once, so you can see the servos work.
void selfTest() {
  const float moves[][2] = {{40, 90}, {140, 90}, {90, 60}, {90, 120}, {90, 90}};
  for (auto &m : moves) {
    panGoal = m[0];
    tiltGoal = m[1];
    Serial.printf("test: P%.0f T%.0f\r\n", panGoal, tiltGoal);
    while (fabs(panGoal - panNow) > 0.5 || fabs(tiltGoal - tiltNow) > 0.5) {
      step();
      delay(5);
    }
    delay(150);
  }
}

void setup() {
  Serial.begin(115200);
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  pan.setPeriodHertz(50);
  tilt.setPeriodHertz(50);
  int a = pan.attach(PAN_PIN, 500, 2400), b = tilt.attach(TILT_PIN, 500, 2400);
  Serial.printf("servos: pan on GPIO %d %s, tilt on GPIO %d %s\r\n", PAN_PIN, a >= 0 ? "ok" : "FAILED",
                TILT_PIN, b >= 0 ? "ok" : "FAILED");
  pan.write(90);
  tilt.write(90);
  lastStep = millis();
  selfTest();
  Serial.println("machine-head ready");
  connectWifi();
}

void loop() {
  connectWifi();
  if (!online && WiFi.status() == WL_CONNECTED) comeOnline();
  if (online && WiFi.status() != WL_CONNECTED) online = false;

  if (online) {
    ArduinoOTA.handle();
    if (server.hasClient()) {
      if (client) client.stop();                     // the newest connection wins
      client = server.accept();
      client.setNoDelay(true);
      client.println("HELLO machine-head 1");
      Serial.printf("client connected from %s\r\n", client.remoteIP().toString().c_str());
      report(client);
    }
    if (client && client.connected()) readLines(client, fromClient, client, "wifi");
  }
  readLines(Serial, fromSerial, Serial, "usb");
  step();
}
