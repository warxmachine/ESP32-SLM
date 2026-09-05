/*
  Tiny character-level RNN running locally on ESP32.

    x_t = onehot(c_t)          // picking a column of Wxh
    h_t = tanh(Wxh x_t + Whh h_{t-1} + bh)
    z_t = Why h_t + by
    P   = softmax(z_t)

  Weights live in flash as int8 + a scale. The math is float.
  Type a prompt in Serial Monitor (115200 baud) and press Enter.
*/

#include <math.h>
#include <string.h>
#include "model_weights.h"

static float temperature = 0.6f;
static int   max_new_chars = 0;
static int   show_why = 0;
static const int TYPE_DELAY_MS = 15;

static float h[HIDDEN_SIZE];
static float hnew[HIDDEN_SIZE];
static float logits[VOCAB_SIZE];
static float probs[VOCAB_SIZE];

static int char_to_ix(char c) {
  unsigned char uc = (unsigned char)c;
  if (uc >= 'A' && uc <= 'Z') uc = (unsigned char)(uc - 'A' + 'a');
  for (int i = 0; i < VOCAB_SIZE; i++) {
    if ((unsigned char)VOCAB[i] == uc) return i;
  }
  return -1;
}

static void softmax_temp(const float *z, float *p, int n, float temp) {
  float maxz = z[0] / temp;
  for (int i = 1; i < n; i++) {
    float v = z[i] / temp;
    if (v > maxz) maxz = v;
  }
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    p[i] = expf(z[i] / temp - maxz);
    sum += p[i];
  }
  if (sum < 1e-8f) sum = 1e-8f;
  for (int i = 0; i < n; i++) p[i] /= sum;
}

static int sample_from(const float *p, int n) {
  float r = (float)esp_random() / 4294967296.0f;
  float acc = 0.0f;
  for (int i = 0; i < n; i++) {
    acc += p[i];
    if (r <= acc) return i;
  }
  return n - 1;
}

/* One-hot times Wxh is just "take column `ix`". Scales stay outside the j loop. */
static void rnn_step(int ix) {
  if (ix < 0 || ix >= VOCAB_SIZE) ix = 0;

  for (int i = 0; i < HIDDEN_SIZE; i++) {
    float acc = 0.0f;
    const int8_t *whh_row = &WHH[i * HIDDEN_SIZE];
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      acc += (float)whh_row[j] * h[j];
    }
    acc = acc * WHH_SCALE;
    acc += (float)BH[i] * BH_SCALE;
    acc += (float)WXH[i * VOCAB_SIZE + ix] * WXH_SCALE;
    hnew[i] = tanhf(acc);
  }

  memcpy(h, hnew, sizeof(h));

  for (int i = 0; i < VOCAB_SIZE; i++) {
    float acc = 0.0f;
    const int8_t *why_row = &WHY[i * HIDDEN_SIZE];
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      acc += (float)why_row[j] * h[j];
    }
    logits[i] = acc * WHY_SCALE + (float)BY[i] * BY_SCALE;
  }
}

static void reset_state() {
  memset(h, 0, sizeof(h));
}

/* Training pairs are `you: ...\nbot: ...`. Without that newline the RNN stays on the question line. */
static int looks_like_you_prompt(const char *p) {
  while (*p == ' ') p++;
  char buf[4];
  for (int i = 0; i < 4; i++) {
    if (!p[i]) return 0;
    unsigned char uc = (unsigned char)p[i];
    if (uc >= 'A' && uc <= 'Z') uc = (unsigned char)(uc - 'A' + 'a');
    buf[i] = (char)uc;
  }
  return buf[0] == 'y' && buf[1] == 'o' && buf[2] == 'u' && buf[3] == ':';
}

static int ends_with_newline(const char *p) {
  if (!p || !*p) return 0;
  while (p[1]) p++;
  return *p == '\n';
}

static void print_why(int picked) {
  int idx[VOCAB_SIZE];
  for (int i = 0; i < VOCAB_SIZE; i++) idx[i] = i;
  for (int a = 0; a < 5; a++) {
    int best = a;
    for (int b = a + 1; b < VOCAB_SIZE; b++) {
      if (probs[idx[b]] > probs[idx[best]]) best = b;
    }
    int tmp = idx[a];
    idx[a] = idx[best];
    idx[best] = tmp;
  }
  Serial.println();
  for (int r = 0; r < 5; r++) {
    int i = idx[r];
    Serial.print("  ");
    Serial.print(r + 1);
    Serial.print(") ");
    if (VOCAB[i] == '\n') Serial.print("\\n");
    else if (VOCAB[i] == ' ') Serial.print("' '");
    else Serial.print(VOCAB[i]);
    Serial.print("  z=");
    Serial.print(logits[i], 2);
    Serial.print("  P=");
    Serial.print(probs[i] * 100.0f, 1);
    Serial.print("%");
    if (i == picked) Serial.print("  <-- picked");
    Serial.println();
  }
}

static void generate(const char *prompt) {
  reset_state();

  int fed = 0;
  int fed_you_nl = 0;
  for (const char *p = prompt; *p; p++) {
    int ix = char_to_ix(*p);
    if (ix < 0) continue;
    rnn_step(ix);
    fed++;
  }
  if (fed == 0) {
    int space_ix = char_to_ix(' ');
    rnn_step(space_ix >= 0 ? space_ix : 0);
  } else if (looks_like_you_prompt(prompt) && !ends_with_newline(prompt)) {
    int nl = char_to_ix('\n');
    if (nl >= 0) {
      rnn_step(nl);
      fed_you_nl = 1;
    }
  }

  while (Serial.available()) Serial.read();

  Serial.print("YOU:   ");
  Serial.println(prompt);
  Serial.print("ESP32: ");
  Serial.print(prompt);
  if (fed_you_nl) Serial.print('\n');

  int blanks = 0;
  for (int n = 0; max_new_chars <= 0 || n < max_new_chars; n++) {
    softmax_temp(logits, probs, VOCAB_SIZE, temperature);
    int ix = sample_from(probs, VOCAB_SIZE);
    Serial.print(VOCAB[ix]);
    if (show_why) print_why(ix);
    if (VOCAB[ix] == '\n') {
      blanks++;
      if (max_new_chars <= 0 && blanks >= 2) break;
    } else {
      blanks = 0;
    }
    if (Serial.available()) break;
    delay(TYPE_DELAY_MS);
    rnn_step(ix);
  }
  Serial.println();
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  delay(1500);

  Serial.println();
  Serial.println("================================");
  Serial.println("  ESP32 TINY LANGUAGE MODEL");
  Serial.print("  params=");
  Serial.print(N_PARAMS);
  Serial.print("  hidden=");
  Serial.print(HIDDEN_SIZE);
  Serial.print("  vocab=");
  Serial.println(VOCAB_SIZE);
  Serial.println("  inference only — no cloud");
  Serial.println("================================");
  Serial.println();

  Serial.print("  temp=");
  Serial.print(temperature, 2);
  Serial.print("  n=");
  if (max_new_chars <= 0) Serial.print("unlimited");
  else Serial.print(max_new_chars);
  Serial.println("  why=off");
  Serial.println("  /temp 0.6    /n 0    /why");
  Serial.println();

  generate("you: what does the rocket do");

  Serial.println("Type a prompt and press Enter.");
  Serial.println();
}

static bool handle_command(const String &line) {
  if (line.startsWith("/temp ")) {
    float t = line.substring(6).toFloat();
    if (t < 0.05f) t = 0.05f;
    if (t > 2.0f) t = 2.0f;
    temperature = t;
    Serial.print("temp = ");
    Serial.println(temperature, 3);
    return true;
  }
  if (line.startsWith("/n ")) {
    int n = line.substring(3).toInt();
    if (n < 0) n = 0;
    max_new_chars = n;
    Serial.print("n = ");
    if (max_new_chars <= 0) Serial.println("unlimited");
    else Serial.println(max_new_chars);
    return true;
  }
  if (line == "/why") {
    show_why = !show_why;
    Serial.println(show_why ? "why = on  (prints z/P after each char)" : "why = off");
    return true;
  }
  return false;
}

void loop() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  if (handle_command(line)) return;

  int why_at = line.indexOf("/why");
  if (why_at >= 0) {
    show_why = 1;
    line.remove(why_at, 4);
    line.trim();
    Serial.println("why = on");
    if (line.length() == 0) return;
  }

  generate(line.c_str());
  Serial.println("Type a prompt and press Enter.");
  Serial.println();
}
