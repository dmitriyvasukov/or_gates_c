#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Обучающие данные: {x1, x2, y} (логическое ИЛИ)
float train[][3] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1},
};

#define train_count (sizeof(train)/sizeof(train[0]))
float eps = 1e-3;
float learning_rate = 1e-1; 

float rand_float(void) {
    return (float)rand() / (float)RAND_MAX;
}

float loss(float w1, float w2, float b) {
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = x1 * w1 + x2 * w2 + b;
        float d = y - train[i][2];
        result += d * d;
    }
    return result / train_count;
}

int main(void) {
    srand(time(0));
    float w1 = rand_float();
    float w2 = rand_float();
    float b = rand_float(); 

    for (size_t i = 0; i < 10000; i++) {
        float l = loss(w1, w2, b);
        if (i % 1000 == 0) {
            printf("Epoch %zu: w1 = %f, w2 = %f, b = %f, loss = %f\n", i, w1, w2, b, l);
        }
        
        // Численное вычисление градиентов
        float dw1 = (loss(w1 + eps, w2, b) - l) / eps;
        float dw2 = (loss(w1, w2 + eps, b) - l) / eps;
        float db = (loss(w1, w2, b + eps) - l) / eps;
        
        // Обновление параметров
        w1 -= learning_rate * dw1;
        w2 -= learning_rate * dw2;
        b -= learning_rate * db;
    }
    
    // Проверка обученной модели
    printf("\nTrained model:\n");
    for (size_t i = 0; i < train_count; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = x1 * w1 + x2 * w2 + b;
        printf("Input: %.1f, %.1f | Output: %.2f | Expected: %.1f\n", 
               x1, x2, y, train[i][2]);
    }
    
    return 0;
}
