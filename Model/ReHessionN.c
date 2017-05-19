//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1200
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define FREE(x) //if (x != NULL) {free(x);}
#define CHECKNULL(x) if (x == NULL) {printf("Memory allocation failed\n"); exit(1);}
#define NRAND next_random = next_random * (unsigned long long)25214903917 + 11;
#define BWRITE(x,f) fwrite(&x , sizeof(real), 1, f);
#define SWRITE(x,f) fprintf(f, "%lf ", x);
#ifdef DEBUG
#define DDMode(f) {f;} 
#else
#define DDMode(f)
#endif

#ifdef GRADCLIP
#define GCLIP(x) ((grad = x), (grad > cur_grad_clip ? cur_grad_clip : (grad < -cur_grad_clip ? -cur_grad_clip : grad)))
#else
#define GCLIP(x) (x)
#endif

#define MINIVALUE 0.00001

#ifdef DROPOUT
#define DROPOUTRATIO 100000
#endif
typedef float real;                    // Precision of double numbers

struct supervision {
  long long function_id;
  long long label;
};

struct training_ins {
  long long id;
  long long c_num;
  long long *cList;
  long long sup_num;
  struct supervision *supList;
};

char train_file[MAX_STRING], test_file[MAX_STRING], val_file[MAX_STRING];
long long  *cCount;
int binary = 1, debug_mode = 2, resample = 20, min_count = 5, num_threads = 1, min_reduce = 1, infer_together = 0, no_lb = 1, no_db = 1, ignore_none = 0, error_log = 0, special_none = 0, printVal = 0 ;
long long c_size = 0, c_length = 100, l_size = 1, l_length = 400, d_size, tot_c_count = 0, NONE_idx = 6;
real lambda1 = 1, lambda2 = 1, lambda3 = 0, lambda4 = 0, lambda5 = 0, lambda6 = 0;
long long ins_num = 225977, ins_count_actual = 0; 
long long test_ins_num = 1900, val_ins_num = 211;
long long iters = 10;
long print_every = 1000;
real alpha = 0.025, starting_alpha, sample = 1e-4;
real grad_clip = 5;
long long useEntropy=1;

struct training_ins * data, *test_ins, *val_ins;
real *c, *l, *d, *cneg, *db, *lb;
real *o;
real ph1, ph2;
real *sigTable, *expTable, *tanhTable;
clock_t start;
#ifdef DROPOUT 
real dropout;
#endif

int negative = 5;
const int table_size = 1e9;
long long *table;

inline void copyIns(struct training_ins *To, struct training_ins *From){
  To->id = From->id;
  To->c_num = From->c_num;
  To->cList = From->cList;
  To->sup_num = From->sup_num;
  To->supList = From->supList;
}

void InitUnigramTable() {
  long long a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table = (long long *)malloc(table_size * sizeof(long long));
  if (table == NULL) {
    fprintf(stderr, "cannot allocate memory for the table\n");
    exit(1);
  }
  for (a = 0; a < c_size; a++) train_words_pow += pow(cCount[a], power);
  i = 0;
  d1 = pow(cCount[a], power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      while (cCount[++i] == 0) continue;
      d1 += pow(cCount[i], power) / (real)train_words_pow;
    }
    if (i >= c_size) i = c_size - 1;
  }
}

// Reads a single word from a file, assuming comma + space + tab + EOL to be word boundaries
// 0: EOF, 1: comma, 2: tab, 3: \n, 4: space
inline int ReadWord(long long *word, FILE *fin) {
  char ch;
  int sgn = 1;
  *word = 0;
  while (!feof(fin)) {
    ch = fgetc(fin);
    switch (ch) {
      case ',':
        return 1;
      case '\t':
        return 2;
      case '\n':
        return 3;
      case ' ':
        return 4;
      case '-':
        sgn = sgn * -1;
        break;
      default:
        *word = *word * 10 + ch - '0';
    }
  }
  *word = *word * sgn;
  return 0;
}

void InitNet() {
  long long a, b;
  a = posix_memalign((void **)&c, 128, (long long)c_size * c_length * sizeof(real));
  CHECKNULL(c)
  a = posix_memalign((void **)&cneg, 128, (long long)c_size * c_length * sizeof(real));
  CHECKNULL(cneg)
  a = posix_memalign((void **)&l, 128, (long long)l_size * l_length * sizeof(real));
  CHECKNULL(l)
  a = posix_memalign((void **)&d, 128, (long long)d_size * l_length * sizeof(real));
  CHECKNULL(d)
  a = posix_memalign((void **)&o, 128, (long long)c_length * l_length * sizeof(real));
  CHECKNULL(o)
  if (0 == no_lb) {
    a = posix_memalign((void **)&lb, 128, (long long)l_size * sizeof(real));
    CHECKNULL(lb)
    for (b = 0; b < l_size; ++b) lb[b] = 0;
  }
  if (0 == no_db) {
    a = posix_memalign((void **)&db, 128, (long long)d_size * sizeof(real));
    CHECKNULL(db)
    for (b = 0; b < d_size; ++b) db[b] = 0;
  }
  cCount = (long long *) calloc(c_size,  sizeof(long long));
  CHECKNULL(cCount)
  memset(cCount, 0, c_size);
  ph2 = 1.0/l_size;
  ph1 = 1 - ph2;
  
  for (b = 0; b < c_size; ++b) for (a = 0; a < c_length; ++a) {
    c[b * c_length + a] = (rand() / (real)RAND_MAX - 0.5) / l_length;
    cneg[b * c_length + a] = 0;
  }
  for (b = 0; b < l_size; ++b) for (a = 0; a < l_length; ++a)
    l[b * l_length + a] = 0;
  for (b = 0; b < d_size; ++b) for (a = 0; a < l_length; ++a)
    d[b * l_length + a] = 0; 
  for (b = 0; b < l_length; ++b) for (a = 0; a < c_length; ++a)
    o[b * c_length + a] = (rand() / (real)RAND_MAX - 0.5) / l_length;
}

void DestroyNet() {
  FREE(lb)
  FREE(db)
  FREE(l)
  FREE(d)
  FREE(c)
  FREE(cneg)
  FREE(o)
  FREE(cCount)
}

void LoadTrainingData(){
  FILE *fin = fopen(train_file, "r");
  if (fin == NULL) {
    fprintf(stderr, "no such file: %s\n", train_file);
    exit(1);
  }
  long long curInsCount = ins_num, a, b;
  
  data = (struct training_ins *) calloc(ins_num, sizeof(struct training_ins));
  while(curInsCount--){
    ReadWord(&data[curInsCount].id, fin);
    ReadWord(&data[curInsCount].c_num, fin);
    ReadWord(&data[curInsCount].sup_num, fin);
    data[curInsCount].cList = (long long *) calloc(data[curInsCount].c_num, sizeof(long long));
    data[curInsCount].supList = (struct supervision *) calloc(data[curInsCount].sup_num, sizeof(struct supervision));

    for (a = data[curInsCount].c_num; a; --a) {
      ReadWord(&b, fin);
      if (b > c_size) c_size = b;
      data[curInsCount].cList[a-1] = b;
    }
    for (a = data[curInsCount].sup_num; a; --a) {
      ReadWord(&b, fin);
      if (b > l_size) l_size = b;
      data[curInsCount].supList[a-1].label = b;
      ReadWord(&b, fin);
      if (b > d_size) d_size = b;
      data[curInsCount].supList[a-1].function_id = b;

      DDMode({printf("(%lld, %lld)", data[curInsCount].supList[a-1].label, data[curInsCount].supList[a-1].function_id);})
    }
    DDMode({printf("\n");})
  }
  c_size++; d_size++; l_size++;
  fclose(fin);
}

void *TrainModelThread(void *id) {
  unsigned long long next_random = (long long)id;
  long long cur_iter = 0, end_id = ((long long)id+1) * ins_num / num_threads;
  long long cur_id, last_id;
  clock_t now;
  long long a, b, i, j, l1, l2 = 0;
  long long update_ins_count = 0, correct_ins = 0, predicted_label = -1;
  real f, g, h;
  #ifdef GRADCLIP
  real grad, cur_grad_clip = grad_clip;
  #endif
  long long target, label;
  struct training_ins *cur_ins;
  real *c_error = (real *) calloc(c_length, sizeof(real));
  real *z = (real *) calloc(l_length, sizeof(real));
  real *z_error = (real *) calloc(l_length, sizeof(real));
  real *score_p = (real *) calloc(l_length, sizeof(real));
  real *score_n = (real *) calloc(l_length, sizeof(real));
  real *sigmoidD = (real *) calloc(l_length, sizeof(real));
  real sum_softmax;
  real *score_kl = (real *) calloc(l_length, sizeof(real));
  struct training_ins tmpIns;

  #ifdef DROPOUT 
  long long dropoutLeft;
  real *c_dropout = (real *) calloc(c_length, sizeof(real));
  real *z_dropout = (real *) calloc(l_length, sizeof(real));
  #endif
  while (cur_iter < iters) {
 
    for (cur_id = (long long)id * ins_num / num_threads; cur_id < end_id - 1; ++cur_id){
      a = end_id - cur_id;
      copyIns(&tmpIns, data + cur_id);
      NRAND
      b = next_random % a;
      copyIns(data + cur_id, data + cur_id + b);
      copyIns(data + cur_id + b, &tmpIns);
    }
    cur_id = (long long)id * ins_num / num_threads;
    last_id = cur_id;
    while(cur_id < end_id){
      // update threads
      if (cur_id - last_id > 1000) {
        ins_count_actual += cur_id - last_id;
        now = clock();
        if (debug_mode > 1) {
          printf("\rAlpha: %f \t Progress: %.2f%% \t Words/thread/sec: %.2fk \t corrected %.2f%% \t on %lld |", alpha,
            ins_count_actual / (real)(ins_num * iters + 1) * 100,
            ins_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000),
            // 100 * (update_ins_count) / ((real)(cur_id - last_id + 1)), 
            100 * correct_ins / ((real) update_ins_count + 1),
            update_ins_count);
          fflush(stdout);
        }
        if (error_log) {
          fprintf(stderr, "\rAlpha: %f \t Progress: %.2f%% \t Words/thread/sec: %.2fk \t corrected %.2f%% \t on %lld |", alpha,
            ins_count_actual / (real)(ins_num * iters + 1) * 100,
            ins_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000),
            // 100 * (update_ins_count) / ((real)(cur_id - last_id + 1)), 
            100 * correct_ins / ((real) update_ins_count + 1),
            update_ins_count);
          fflush(stderr);
        }
        last_id = cur_id;
        update_ins_count = 0;
        correct_ins = 0;
        alpha = starting_alpha * (1 - ins_count_actual / (real) (ins_num * iters + 1));
        if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
      }

      cur_ins = data + cur_id;
      
      DDMode ({
      printf("curid: %lld, %lld\n", cur_id, cur_ins->id);
      for (i = 0; i < cur_ins->sup_num; ++i) 
        printf("(%lld, %lld)", cur_ins->supList[i].function_id, cur_ins->supList[i].label);
      printf("\n");
      })
   
      // feature embedding learning
      for (i = 0; i < resample; ++i){
           b = -1;
        while(b < 0){
          if (b != -2 && sample > 0) {
            //down sampling
            NRAND
            b = next_random % cur_ins->c_num;
            b = cur_ins->cList[b];
   
            real ran = (sqrt(cCount[b] / (sample * tot_c_count)) + 1) * (sample * tot_c_count) / cCount[b];
            NRAND
            if (ran < (next_random & 0xFFFF) / (real)65536) b = -2;
          } else {
            NRAND
            b = next_random % cur_ins->c_num;
            b = cur_ins->cList[b];
          }
        }
        l1 = b * c_length;
        for (a = 0; a < c_length; ++a) c_error[a] = 0.0;
        for (j = 0; j < negative + 1; ++j){
          NRAND
          if (0 == j){
            target = next_random % cur_ins->c_num;
            target = cur_ins->cList[target];
            label = 1;
          } else {
            target = table[next_random % table_size];
            label = 0;
          }
          if (target == b) continue;
          l2 = target * c_length;
          f = 0.0;
          for (a = 0; a < c_length; ++a) f += c[a + l1] * cneg[a + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha * lambda1;
          else if (f < -MAX_EXP) g = (label - 0) * alpha * lambda1;
          else {
            g = (label - sigTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha * lambda1;
          }
          for (a = 0; a < c_length; ++a) c_error[a] += g * cneg[a + l2];
          for (a = 0; a < c_length; ++a) cneg[a + l2] += GCLIP(g * c[a + l1]);
        }
        for (a = 0; a < c_length; ++a) c[a + l1] += GCLIP(c_error[a]);
      }
      for (a = 0; a < c_length; ++a) c_error[a] = 0.0;

#ifdef DROPOUT
      dropoutLeft = 0;
      for (i = 0; i < cur_ins->c_num; ++i) {
        NRAND
        if (next_random % DROPOUTRATIO >= dropout) { //notdropout
          dropoutLeft += 1;
          l1 = cur_ins->cList[i] * c_length;
          for (j = 0; j < c_length; ++j) c_error[j] += c[l1 + j];
        } else {
          cur_ins->cList[i] = (-1 * cur_ins->cList[i]) - 1;
        }
      }
      for (a = 0; a < c_length; ++a) c_error[a] = (c_error[a] + MINIVALUE) / (dropoutLeft + MINIVALUE);
#else
      for (i = 0; i < cur_ins->c_num; ++i) {
        l1 = cur_ins->cList[i] * c_length;
        for (j = 0; j < c_length; ++j) c_error[j] += c[l1 + j];
      }
      for (a = 0; a < c_length; ++a) c_error[a] /= i;
#endif

#ifdef DROPOUT
      for (a = 0; a < c_length; ++a) {
        NRAND
        if (next_random % DROPOUTRATIO < dropout) { //dropout
          c_error[a] = 0;
          c_dropout[a] = 1;
        } else {
          c_dropout[a] = 0;
        }
      }
#endif

      for (a = 0; a < l_length; ++a) {
        f = 0;
#ifdef DROPOUT
        NRAND
        if (next_random % DROPOUTRATIO < dropout) { //dropout
          z_dropout[a] = 1;
          z[a] = 0;
          continue;
        } else {
          z_dropout[a] = 0;
        }
#endif
        l1 = a * c_length;
        for (i = 0; i < c_length; ++i) f += c_error[i] * o[l1 + i];
#ifdef ACTIVE
        if (f < -MAX_EXP) z[a] = -1;
        else if (f > MAX_EXP) z[a] = 1;
        else z[a] = tanhTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
#else
        z[a] = f;
#endif
      }
      for (a = 0; a < l_length; ++a) z_error[a] = 0;
      // infer true labels
      // score here is prob, which is the larger the better
      for (i = 0; i < l_size; ++i) {
        score_n[i] = 0;
        score_p[i] = 0;
      }
      for (i = 0 ; i < cur_ins->sup_num ; ++i){
        j = cur_ins->supList[i].function_id;
        l1 = j * l_length;
        if (0 == no_db) f = db[j];
        else f = 0;
        for (a = 0; a < l_length; ++a) f+= z[a] * d[l1 + a];
        if (f > MAX_EXP) g = 1.0/(1.0 + exp(-f));
        else if (f < -MAX_EXP) g = 1.0/(1.0 + exp(-f));
        else g = sigTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
        a = cur_ins->supList[i].label;
        sigmoidD[a] = g;
        score_p[a] += log(g * ph1 + (1 - g) * ph2);
        score_n[a] += log(g * (1 - ph1) + (1 - g) * (1 - ph2));
        z_error[a] = 1;
        DDMode({printf("(%f, %f, %f, %f, %f),", ph1, ph2, g, g * ph1 + (1 - g) * ph2, g * (1 - ph1) + (1 - g) * (1 - ph2));})
      }
      f = 0.0; for (i = 0; i < l_size; ++i) f += score_n[i];
      g = -INFINITY;
      label = -1;
      for (i = 0; i < l_size; ++i) if ((z_error[i] > 0 ) && (0 == ignore_none || i != NONE_idx)) {
        h = f - score_n[i] + score_p[i];
        if (h > g){
          label = i;
          g = h;
        }
      }
      DDMode({printf("\n");})
      if (0 != ignore_none && -1 == label) {
#ifdef DROPOUT
        for (i = 0; i < cur_ins->c_num; ++i) {
          if (cur_ins->cList[i] < 0) {
            cur_ins->cList[i] = -1 * (cur_ins->cList[i] + 1);
          }
        }
#endif
        ++cur_id;

        continue;
      }
      if(-1 == label){
        for (i = 0; i < l_size; ++i) printf("-1: %lld, %f\n", i, score_p[i]);
        exit(1);
      }
      if(debug_mode > 2){
        printf("%lld, %lld:", label, cur_ins->sup_num);
        for (i = 0; i < cur_ins->sup_num; ++i){
          printf("(%lld, %lld);", cur_ins->supList[i].function_id, cur_ins->supList[i].label);
        }
        putchar('\n');
      }

      // reini z_error;
      for (a = 0; a < l_length; ++a) z_error[a] = 0;
      // update params 
      
      //update predicted label && predicton model
      //updadte predicted label
      sum_softmax = 0.0;
      g = -INFINITY; predicted_label = -1;
      for (i = 0 ; i < l_size; ++i) {
        if (0 == no_lb) f = lb[i];
        else f = 0;
        l1 = i * l_length;
        for (a = 0; a < l_length; ++a) f += z[a] * l[l1 + a];
        DDMode({printf("(%f, %lld), ", f, i);})
        score_kl[i] = f;
        if (f > g) {
          g = f;
          predicted_label = i;
        }
      }
      for (i = 0; i < l_size; ++i) {
        f = score_kl[i] - g;
        if (debug_mode > 2) printf("f: %f, %f, %f\n", f, g, score_kl[i]);
        if (f < -MAX_EXP) score_kl[i] = 0;
        else if (f > MAX_EXP) printf("error! softmax over 1!\n");
        else score_kl[i] = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
        sum_softmax += score_kl[i];
      }
      if (debug_mode > 2) printf("softmax: %f, %f\n", sum_softmax, g);
      // update l, lb
      if (!special_none || NONE_idx != label) {
        l1 = label * l_length;
        f = alpha * score_kl[label] / sum_softmax;
        if (debug_mode > 2) printf("%f, %f, %f, %f\n",l[l1], z[0], z_error[0], f);
        if (0 == no_lb) lb[label] += GCLIP(alpha - f - lambda3 * lb[label]);
        for (a = 0; a < l_length; ++a)
#ifdef DROPOUT
            if(0 ==z_dropout[a])
#endif 
        {
          z_error[a] += l[l1 + a] * (alpha - f);
          l[l1 + a] += GCLIP(z[a] * (alpha - f) - lambda3 * l[l1 + a]);
        }
        for (i = 0; i < l_size; ++i) if (i != label) {  
          l1 = i * l_length;
          f = alpha * score_kl[i] / sum_softmax;
          if (0 == no_lb) lb[i] -= GCLIP(f + lambda3 * lb[i]);
          for (a = 0; a < l_length; ++a) 
#ifdef DROPOUT
            if(0 ==z_dropout[a])
#endif 
          {
            z_error[a] -= l[l1 + a] * f;
            l[l1 + a] -= GCLIP(z[a] * f + lambda3 * l[l1 + a]);
          }
        }
      } else {
        g = alpha / (l_size - 1);
        for (i = 0; i < l_size; ++i) if (i != NONE_idx) {
          f = g - score_kl[i] / sum_softmax;
          if (0 == no_lb) lb[i] += GCLIP(g - lambda3 * lb[i]);
          l1 = i * l_length;
          for (a = 0; a < l_length; ++a){
            z_error[a] += l[l1 + a] * f;
            l[l1 + a] += GCLIP(z[a] * f - lambda3 * l[l1 + a]);
          }
        }
      }
      if (debug_mode > 2) printf("1:%f, %f, %f, %f, %f, %f, %f\n", z_error[0], o[0], z[0], l[0], f, score_kl[label], sum_softmax);

      DDMode({printf("label: %lld, predicted: %lld\n", label, predicted_label);})
      correct_ins += (label == predicted_label) && (label != NONE_idx);
      update_ins_count += (label != NONE_idx);
      // update d, db
      // update ph1, ph2
      for (i = 0 ; i < cur_ins->sup_num ; ++i){
        j = cur_ins->supList[i].function_id;
        a = cur_ins->supList[i].label;
        f = sigmoidD[a] * ph1 + (1 - sigmoidD[a]) * ph2;
        if (debug_mode > 2) printf("%lld, %lld, %f, %f, %f, %f, %f \n", j, a, f, sigmoidD[a], ph1, ph2, sigmoidD[a] * ph1 + (1 - sigmoidD[a]) * ph2);
        if (a == label) {
          //d, db
          g = alpha * lambda2 * (ph1 - ph2) * sigmoidD[a] * (1- sigmoidD[a]) / f;
          l1 = j * l_length;
          if (0 == no_db) db[j] += GCLIP(g - alpha * lambda4 * db[j]);
          for (b = 0; b < l_length; ++b)
#ifdef DROPOUT
            if(0 ==z_dropout[b])
#endif 
          {
            z_error[b] += d[l1 + b] * g;
            d[l1 + b] += GCLIP(z[b] * g - alpha * lambda4 * d[l1 + b]);
          }
        } else {
          //d, db
          g = alpha * lambda2 * (ph2 - ph1) * sigmoidD[a] * (1 - sigmoidD[a]) / f;
          l1 = j * l_length;
          if (0== no_db) db[j] += GCLIP(g - alpha * lambda4 * db[j]);
          for (b = 0; b< l_length; ++b) 
#ifdef DROPOUT
            if(0 ==z_dropout[b])
#endif 
          {
            z_error[b] += d[l1 + b] * g;
            d[l1 + b] += GCLIP(z[b] * g - alpha * lambda4 * d[l1 + b]);
          }
        }
      }
#ifdef ACTIVE
      for (a = 0; a < l_length; ++a) {
        z_error[a] *= (1 - z[a]*z[a]);
      }
#endif      
      // update o
      if (debug_mode > 2) printf("2:%f, %f\n", z_error[0], o[0]);
      for (a = 0; a < l_length; ++a) 
#ifdef DROPOUT
        if(0 ==z_dropout[a])
#endif 
      {
        l1 = a * c_length;
        for (b = 0; b < c_length; ++b) o[l1 + b] += GCLIP(z_error[a] * c_error[b] - alpha * lambda5 * o[l1 + b]);
      }
      for (a = 0; a < c_length; ++a) c_error[a] = 0;
      for (a = 0; a < l_length; ++a) {
        l1 = a * c_length;
        for (b = 0; b < c_length; ++b) c_error[b] += z_error[a] * o[l1 + b];
      }
#ifdef DROPOUT
      for (a = 0; a < c_length; ++a) {
        if (0==c_dropout[a]) 
          c_error[a] = (c_error[a] + MINIVALUE)/(dropoutLeft + MINIVALUE);
        else
          c_error[a] = 0;
      }  
      for (i = 0; i < cur_ins->c_num; ++i) {
        if (cur_ins->cList[i] >= 0) {
          l1 = cur_ins->cList[i] * c_length;
          for (j = 0; j < c_length; ++j) if(0==c_dropout[j])
            c[l1 + j] += GCLIP(c_error[j]- alpha * lambda6 * c[l1 + j]);
        } else {
          cur_ins->cList[i] = -1 * (cur_ins->cList[i] + 1);
        }
      }
#else
      for (a = 0; a < c_length; ++a) c_error[a] /= cur_ins->c_num;
      for (i = 0; i < cur_ins->c_num; ++i) {
        l1 = cur_ins->cList[i] * c_length;
        for (j = 0; j < c_length; ++j) c[l1 + j] += GCLIP(c_error[j]- alpha * lambda6 * c[l1 + j]);
      }
#endif
      // update index
      ++cur_id;
    }
    ++cur_iter;
  }
  pthread_exit(NULL);
}

void TrainModel() {
  struct training_ins tmpIns;
  long long a, b;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  if (pt == NULL) {
    fprintf(stderr, "cannot allocate memory for threads\n");
    exit(1);
  }
  starting_alpha = alpha;
  tot_c_count = 0;
  memset(cCount, 0, c_size);
  if (debug_mode > 1) printf("shuffling and building sub-sampling table\n");
  if (negative > 0) {
    for (a = 0; a < ins_num; ++a) {
      //shuffle
      b = ((int)(rand() / (RAND_MAX / ins_num))) % ins_num;

      copyIns(&tmpIns, data + a);
      copyIns(data + a, data + b);
      copyIns(data + b, &tmpIns);
    }
    for (a = 0; a < ins_num; ++a) {
      //count
      for (b = data[a].c_num; b; --b) {
        ++cCount[data[a].cList[b-1]];
        ++tot_c_count;
      }
    }
    InitUnigramTable();
  } else {
    for (a = 0; a < ins_num; ++a) {
      //shuffle
      b = ((int)(rand() / (RAND_MAX / ins_num))) % ins_num;
      copyIns(&tmpIns, data + a);
      copyIns(data + a, data + a + b);
      copyIns(data + a + b, &tmpIns);
    }
  }
  if (debug_mode > 1) printf("Starting training using threads %d\n", num_threads);
  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  free(table);
  free(pt);
}

real calculateEntropy(real *tmp_predict_scores){
  real f, g = -INFINITY, sum_softmax = 0;
  long long i;
  for (i = 0; i < l_size; ++i) if (i != NONE_idx){
    g = g > tmp_predict_scores[i] ? g : tmp_predict_scores[i];
  }
  for (i = 0; i < l_size; ++i) if (i != NONE_idx){
    f = tmp_predict_scores[i] - g;
    if (f < -MAX_EXP) tmp_predict_scores[i] = exp(f);
    else if (f > MAX_EXP) printf("error! softmax over 1!\n");
    else tmp_predict_scores[i] = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    sum_softmax += tmp_predict_scores[i];
  }
  f = 0;
  for (i = 0; i < l_size; ++i) if (i != NONE_idx){
    g = tmp_predict_scores[i] / sum_softmax;
    f -= g * log(g);
  }
  return f;
}

real calculateInnerProd(real *tmp_predict_scores){
  long long i;
  real g = -INFINITY;
  for (i = 0; i < l_size; ++i) if (i != NONE_idx){
    g = g > tmp_predict_scores[i] ? g : tmp_predict_scores[i];
  }
  return -g;
}

void EvaluateModel() {
  long long i, j, a, b;
  long long l1;
  real f, g;
  real *cs = (real *) calloc(c_length, sizeof(real));
  real *z = (real *) calloc(l_length, sizeof(real));
  if (0 != ignore_none) {
    long long correct = 0;
    long long act_ins_num = 0;
    for (i = 0; i < val_ins_num; ++i){
      struct training_ins * cur_ins = val_ins + i;
      //calculate z;
      if (cur_ins->supList[0].label == NONE_idx)
        continue;
      for (j = 0; j < c_length; ++j)
        cs[j] = 0;
      for (a = 0; a < cur_ins->c_num; ++a) {
        l1 = c_length * cur_ins->cList[a];
        for (j = 0; j < c_length; ++j) cs[j] += c[l1 + j];
      }
      for (j = 0; j < c_length; ++j) cs[j] /= cur_ins->c_num;
      for (a = 0; a < l_length; ++a){
        g = 0;
        l1 = a * c_length;
        for (j = 0; j < c_length; ++j) g += cs[j] * o[l1 + j];
#ifdef ACTIVE
        if (g < -MAX_EXP) z[a] = -1;
        else if (g > MAX_EXP) z[a] = 1;
        else z[a] = tanhTable[(int)((g + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
#else
        z[a] = g;
#endif
      }
      b = -1; g = 0;
      for (j = 0; j < l_size; ++j) if (j != NONE_idx) {
        if (0 == no_lb) f = lb[j];
        else f = 0;
        l1 = j * l_length;
        for (a = 0; a < l_length; ++a) f += z[a] * l[l1 + a];
        if (-1 == b || f > g){
          g = f;
          b = j;
        }
        if (debug_mode > 2) printf("%f, ", f);
      }
      if (debug_mode > 2) printf("%lld, %lld, %lld\n", i, cur_ins->supList[0].label, b);
      correct += (b == cur_ins->supList[0].label);
      ++act_ins_num;
    }
    if (0 == printVal) printf("%f, ", (real) correct / act_ins_num * 100);
    correct = 0;
    act_ins_num = 0;
    for (i = 0; i < test_ins_num; ++i){
      struct training_ins * cur_ins = test_ins + i;
      //calculate z;
      if (cur_ins->supList[0].label == NONE_idx)
        continue;
      for (j = 0; j < c_length; ++j)
        cs[j] = 0;
      for (a = 0; a < cur_ins->c_num; ++a) {
        l1 = c_length * cur_ins->cList[a];
        for (j = 0; j < c_length; ++j) cs[j] += c[l1 + j];
      }
      for (j = 0; j < c_length; ++j) cs[j] /= cur_ins->c_num;
      for (a = 0; a < l_length; ++a){
        g = 0;
        l1 = a * c_length;
        for (j = 0; j < c_length; ++j) g += cs[j] * o[l1 + j];
#ifdef ACTIVE
        if (g < -MAX_EXP) z[a] = -1;
        else if (g > MAX_EXP) z[a] = 1;
        else z[a] = tanhTable[(int)((g + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
#else
        z[a] = g;
#endif
      }
      b = -1; g = 0;
      for (j = 0; j < l_size; ++j) if (j != NONE_idx) {
        if (0 == no_lb) f = lb[j];
        else f = 0;
        l1 = j * l_length;
        for (a = 0; a < l_length; ++a) f += z[a] * l[l1 + a];
        if (-1 == b || f > g){
          g = f;
          b = j;
        }
        if (debug_mode > 2) printf("%f, ", f);
      }
      if (printVal) printf("%lld, %lld\n", cur_ins->supList[0].label, b);
      if (debug_mode > 2) printf("%lld, %lld, %lld\n", i, cur_ins->supList[0].label, b);
      correct += (b == cur_ins->supList[0].label);
      ++act_ins_num;
    }
    if (0 == printVal) printf("%f\n", (real) correct / act_ins_num * 100);
  } else {
    long long correct = 0;
    long long act_ins_num = 0, act_pred_num = 0;
    real *entropy_list = (real *) calloc(val_ins_num, sizeof(real));
    long long *label_list = (long long *) calloc(val_ins_num, sizeof(long long));
    real *predict_scores = (real *) calloc(l_size, sizeof(real));

    for (i = 0; i < val_ins_num; ++i){
      struct training_ins * cur_ins = val_ins + i;
      //calculate z;
      act_ins_num += (cur_ins->supList[0].label == NONE_idx ? 0 : 1);
      for (j = 0; j < c_length; ++j)
        cs[j] = 0;
      for (a = 0; a < cur_ins->c_num; ++a) {
        l1 = c_length * cur_ins->cList[a];
        for (j = 0; j < c_length; ++j) cs[j] += c[l1 + j];
      }
      for (j = 0; j < c_length; ++j) cs[j] /= cur_ins->c_num;
      for (a = 0; a < l_length; ++a){
        g = 0;
        l1 = a * c_length;
        for (j = 0; j < c_length; ++j) g += cs[j] * o[l1 + j];
#ifdef ACTIVE
        if (g < -MAX_EXP) z[a] = -1;
        else if (g > MAX_EXP) z[a] = 1;
        else z[a] = tanhTable[(int)((g + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
#else
        z[a] = g;
#endif
      }
      b = -1; g = 0;
      for (j = 0; j < l_size; ++j) {
        if (0 == no_lb) f = lb[j];
        else f = 0;
        l1 = j * l_length;
        for (a = 0; a < l_length; ++a) f += z[a] * l[l1 + a];
        if (-1 == b || f > g){
          g = f;
          b = j;
        }
        predict_scores[j] = f;
      }
      label_list[i] = (b==cur_ins->supList[0].label) && (NONE_idx != b);
      if (NONE_idx != b) {
        if (useEntropy) entropy_list[i] = calculateEntropy(predict_scores);
        else entropy_list[i] = calculateInnerProd(predict_scores);
      } else {
        entropy_list[i] = INFINITY;
      }
    }
    real min_entropy = INFINITY, max_entropy = -INFINITY, best_pre = -INFINITY, best_rec = -INFINITY, best_f1 = -INFINITY, best_threshold = 1;

    for (i = 0; i < val_ins_num; ++i) if (entropy_list[i] < INFINITY) {
      min_entropy = min_entropy < entropy_list[i] ? min_entropy : entropy_list[i];
      max_entropy = max_entropy > entropy_list[i] ? max_entropy : entropy_list[i]; 
    }
    max_entropy = (max_entropy - min_entropy)/100;
    for (a = 1; a < 100; ++a) {
      correct = 0;
      act_pred_num = 0;
      f = min_entropy + max_entropy * a;
      for (i = 0; i < val_ins_num; ++i){
        if (entropy_list[i] < f) {
          correct += label_list[i];
          ++act_pred_num;
        }
      }
      if ((real) 200 * correct / (act_pred_num + act_ins_num) > best_f1) {
        best_f1 = (real) 200 * correct / (act_pred_num + act_ins_num);
        best_pre = 100*(correct+MINIVALUE)/(act_pred_num+MINIVALUE);
        best_rec = 100*(correct+MINIVALUE)/(act_ins_num+MINIVALUE);
        best_threshold = f;
      }
    }

    if (0 == printVal) printf("%f,%f,%f,", best_pre, best_rec, best_f1);

    if (printVal) printf("\n");

    correct = 0;
    act_pred_num = 0;
    act_ins_num = 0;
    for (i = 0; i < test_ins_num; ++i){
      struct training_ins * cur_ins = test_ins + i;
      //calculate z;
      act_ins_num += (cur_ins->supList[0].label == NONE_idx ? 0 : 1);
      for (j = 0; j < c_length; ++j)
        cs[j] = 0;
      for (a = 0; a < cur_ins->c_num; ++a) {
        l1 = c_length * cur_ins->cList[a];
        for (j = 0; j < c_length; ++j) cs[j] += c[l1 + j];
      }
      for (j = 0; j < c_length; ++j) cs[j] /= cur_ins->c_num;
      for (a = 0; a < l_length; ++a){
        g = 0;
        l1 = a * c_length;
        for (j = 0; j < c_length; ++j) g += cs[j] * o[l1 + j];
#ifdef ACTIVE
        if (g < -MAX_EXP) z[a] = -1;
        else if (g > MAX_EXP) z[a] = 1;
        else z[a] = tanhTable[(int)((g + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
#else
        z[a] = g;
#endif
      }
      // l2 = i * l_size;
      b = -1; g = 0;
      for (j = 0; j < l_size; ++j) {
        if (0 == no_lb) f = lb[j];
        else f = 0;
        l1 = j * l_length;
        for (a = 0; a < l_length; ++a) f += z[a] * l[l1 + a];
        if (-1 == b || f > g){
          g = f;
          b = j;
        }
        predict_scores[j] = f;
      }
      if(printVal) printf("%f, ", g);

      if (useEntropy) g = calculateEntropy(predict_scores);
      else g = calculateInnerProd(predict_scores);
    
      if (printVal) printf("%f, %f, %d, %lld, %lld\n", g, best_threshold, g < best_threshold, b, cur_ins->supList[0].label);
      if (g < best_threshold && NONE_idx != b) {
        correct += (b == cur_ins->supList[0].label);
        ++act_pred_num;
      }
    }
    if (0 == printVal) printf("%f,%f,%f\n", (real)100*(correct+MINIVALUE)/(act_pred_num+MINIVALUE), (real)100*(correct+MINIVALUE)/(act_ins_num+MINIVALUE), (real)200*correct/(act_ins_num+act_pred_num));
  }
  FREE(cs);
  FREE(z);
}

void LoadTestingData(){
  FILE *fin = fopen(test_file, "r");
  if (fin == NULL) {
    fprintf(stderr, "no such file: %s\n", test_file);
    exit(1);
  }
  if (debug_mode > 1) printf("curInsCount: %lld\n", test_ins_num);
  long long curInsCount = test_ins_num, a, b;
  
  test_ins = (struct training_ins *) calloc(test_ins_num, sizeof(struct training_ins));
  while(curInsCount--){
    test_ins[curInsCount].id = 1;
    ReadWord(&test_ins[curInsCount].id, fin);
    ReadWord(&test_ins[curInsCount].c_num, fin);
    ReadWord(&test_ins[curInsCount].sup_num, fin);
    test_ins[curInsCount].cList = (long long *) calloc(test_ins[curInsCount].c_num, sizeof(long long));
    test_ins[curInsCount].supList = (struct supervision *) calloc(test_ins[curInsCount].sup_num, sizeof(struct supervision));
 
    for (a = test_ins[curInsCount].c_num; a; --a) {
      ReadWord(&b, fin);
      test_ins[curInsCount].cList[a-1] = b;
    }
    for (a = test_ins[curInsCount].sup_num; a; --a) {
      ReadWord(&b, fin);
      test_ins[curInsCount].supList[a-1].label = b;
      ReadWord(&b, fin);
      test_ins[curInsCount].supList[a-1].function_id = b;
    }
  }
  if ((debug_mode > 1)) {
    printf("load Done\n");
    printf("c_size: %lld, d_size: %lld, l_size: %lld\n", c_size, d_size, l_size);
  }
  fclose(fin);
 }

void LoadValidationData(){
  FILE *fin = fopen(val_file, "r");
  if (fin == NULL) {
    fprintf(stderr, "no such file: %s\n", val_file);
    exit(1);
  }
  if (debug_mode > 1) printf("curInsCount: %lld\n", val_ins_num);
  long long curInsCount = val_ins_num, a, b;
 
  val_ins = (struct training_ins *) calloc(val_ins_num, sizeof(struct training_ins));
  while(curInsCount--){
    val_ins[curInsCount].id = 1;
    ReadWord(&val_ins[curInsCount].id, fin);
    ReadWord(&val_ins[curInsCount].c_num, fin);
    ReadWord(&val_ins[curInsCount].sup_num, fin);
    val_ins[curInsCount].cList = (long long *) calloc(val_ins[curInsCount].c_num, sizeof(long long));
    val_ins[curInsCount].supList = (struct supervision *) calloc(val_ins[curInsCount].sup_num, sizeof(struct supervision));
 
    for (a = val_ins[curInsCount].c_num; a; --a) {
      ReadWord(&b, fin);
      val_ins[curInsCount].cList[a-1] = b;
    }
    for (a = val_ins[curInsCount].sup_num; a; --a) {
      ReadWord(&b, fin);
      val_ins[curInsCount].supList[a-1].label = b;
      ReadWord(&b, fin);
      val_ins[curInsCount].supList[a-1].function_id = b;
    }
  }
  if ((debug_mode > 1)) {
    printf("load Done\n");
    printf("c_size: %lld, d_size: %lld, l_size: %lld\n", c_size, d_size, l_size);
  }
  fclose(fin);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  srand(19940410);
  int i;
  if (argc == 1) {
    printf("ReHession alpha 1.0\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("-cleng\n-lleng\n-train\n-debug\n-binary\n-alpha\n-resample\n-sample\n-negative\n-threads\n-min-count\n-instances\n-infer_together\n-alpha_update_every\n-iter\n-none_idx\n-no_lb\n-no_db\n-lambda1\n-lambda2\n-grad_clip\n-ingore_none\n-error_log\n-normL\n-dropout(D Mode)\nlambda1: skip-gram\nlambda2: truth finding\nlambda3: l\nlambda4: d\nlambda5: o\n lambda6: c\n");
    printf("\nExamples:\n");
    printf("./rmodify -train /shared/data/ll2/CoType/data/intermediate/KBP/train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/test.data -threads 20 -NONE_idx 6 -cleng 30 -lleng 50 -resample 30 -ignore_none 0 -iter 100 -normL 0 -debug 2 -dropout 0.5\n\n");//-none_idx 5 
    return 0;
  }
  test_file[0] = 0;
  val_file[0] = 0;
  train_file[0] = 0;
  if ((i = ArgPos((char *)"-cleng", argc, argv)) > 0) c_length = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-lleng", argc, argv)) > 0) l_length = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-special_none", argc, argv)) > 0) special_none = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-useEntropy", argc, argv)) > 0) useEntropy = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-test", argc, argv)) > 0) strcpy(test_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-val", argc, argv)) > 0) strcpy(val_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-resample", argc, argv)) > 0) resample = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-instances", argc, argv)) > 0) ins_num = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-test_instances", argc, argv)) > 0) test_ins_num = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-val_instances", argc, argv)) > 0) val_ins_num = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-infer_together", argc, argv)) > 0) infer_together = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha_update_every", argc, argv)) > 0) print_every = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-print_val", argc, argv)) > 0) printVal = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iters = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-none_idx", argc, argv)) > 0) NONE_idx = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-no_lb", argc, argv)) > 0) no_lb = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-no_db", argc, argv)) > 0) no_db = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-ignore_none", argc, argv)) > 0) ignore_none = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-lambda1", argc, argv)) > 0) lambda1 = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-lambda2", argc, argv)) > 0) lambda2 = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-lambda3", argc, argv)) > 0) lambda3 = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-lambda4", argc, argv)) > 0) lambda4 = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-lambda5", argc, argv)) > 0) lambda5 = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-lambda6", argc, argv)) > 0) lambda6 = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-grad_clip", argc, argv)) > 0) grad_clip = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-error_log", argc, argv)) > 0) error_log = atoi(argv[i + 1]);
#ifdef DROPOUT
  if ((i = ArgPos((char *)"-dropout", argc, argv)) > 0) dropout = atof(argv[i + 1]) * DROPOUTRATIO;
#endif
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  sigTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  tanhTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  if (sigTable == NULL) {
    fprintf(stderr, "out of memory\n");
    exit(1);
  }
  if (debug_mode > 1) printf("Starting\n");
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    sigTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    tanhTable[i] = (expTable[i] * expTable[i] - 1)/(expTable[i] * expTable[i] + 1);
  }

  if (debug_mode > 1) printf("Loading training file %s\n", train_file);
  LoadTrainingData();
  if (debug_mode > 1) printf("Initialization\n");
  InitNet();
  if (debug_mode > 1) printf("start training, iters: %lld \n ", iters);
  TrainModel();
  if (debug_mode > 1) printf("\nLoading test file %s\n", test_file);
  LoadTestingData();
  if (debug_mode > 1) printf("\nLoading validation file %s\n", val_file);
  LoadValidationData();
  if (debug_mode > 1) printf("start Testing \n ");
  EvaluateModel();
  if (debug_mode > 1) printf("releasing memory\n");
  DestroyNet();
  free(expTable);
  free(sigTable);
  free(tanhTable);
  return 0;
}
