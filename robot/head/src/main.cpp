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

float panNow = 90, tiltNow = 90, panGoal = 90, tiltGoal = 90, speed = SPEED;
unsigned long lastStep = 0, lastWifiTry = 0;
bool online = false;

void report(Print &out) {
  out.printf("OK P%.0f T%.0f\n", panGoal, tiltGoal);
}

void handle(String cmd, Print &out) {
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
  report(out);
}

void readLines(Stream &in, String &buf, Print &out) {
  while (in.available()) {
    char c = in.read();
    if (c == '\n' || c == '\r') {
      handle(buf, out);
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
  Serial.printf("wifi: joining %s\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.setHostname(HOSTNAME);
  WiFi.begin(WIFI_SSID, WIFI_PASS, WIFI_CHANNEL);
#endif
}

void comeOnline() {
  online = true;
  Serial.printf("wifi: %s, listening on port %d\n", WiFi.localIP().toString().c_str(), PORT);
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
}

void setup() {
  Serial.begin(115200);
  pan.attach(PAN_PIN, 500, 2400);
  tilt.attach(TILT_PIN, 500, 2400);
  pan.write(90);
  tilt.write(90);
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
      report(client);
    }
    if (client && client.connected()) readLines(client, fromClient, client);
  }
  readLines(Serial, fromSerial, Serial);
  step();
}
