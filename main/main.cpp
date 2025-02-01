#include <stdio.h>
#include <esp_log.h>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "palm_detection_model.h"
// #include "hand_landmark_model.h"

typedef struct
{
    const tflite::Model* model;
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input;
    TfLiteTensor* output;
} model_info;

model_info get_model(const unsigned char* model_buf){
    ESP_LOGI("get_model", "Started @ get_model");
    printf("Started @ get_model\n");
    const int kTensorArenaSize =  3500000;
    uint8_t tensor_arena[kTensorArenaSize];

    // 1. load palm detection model
    const tflite::Model* model = tflite::GetModel(model_buf);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        printf("Model provided is schema version %d not equal to supported version %d\n",
               (int) model->version(), TFLITE_SCHEMA_VERSION);
        return {nullptr, nullptr, nullptr, nullptr};
    }
    printf("Loaded model\n");
    ESP_LOGI("get_model", "Loaded model");
    // --------------- PALM DETECTION MODEL ------------------
    // Input: tensor: float32[1,192,192,3]
    // Ouputs:
    // 1. detection_boxes: float32[1,2016,18]
    // 2. detection_scores: float32[1,2016,1]

    // Add all oprations to the resolver
    /* PalmDetectionModel Operations: 
    1.	Prelu (PReLU)
	2.	DepthwiseConv2D
	3.	Conv2D
	4.	Add
	5.	Dequantize
	6.	MaxPool2D
	7.	Pad
    8.  Resize Bilinear
    9.  Reshape
    */
    static tflite::MicroMutableOpResolver<2> resolver;
    bool resolver_failure = false;
    #define is_ok(func) if (func != kTfLiteOk) { resolver_failure = true ; }

    is_ok(resolver.AddPrelu());
    is_ok(resolver.AddDepthwiseConv2D());
    is_ok(resolver.AddConv2D());
    is_ok(resolver.AddAdd());
    is_ok(resolver.AddDequantize());
    is_ok(resolver.AddMaxPool2D());
    is_ok(resolver.AddPad());
    is_ok(resolver.AddResizeBilinear());
    is_ok(resolver.AddReshape());

    if (resolver_failure)
    {
        printf("Failed to add operations to resolver\n");
        return {nullptr, nullptr, nullptr, nullptr};
    }
    printf("Added all operations to resolver\n");

    // Build an interpreter to run the model with. We need to log the arena size to better allocate it in the future
    tflite::MicroInterpreter* interpreter = new tflite::MicroInterpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    printf("Created Interpreter\n");
    ESP_LOGI("get_model", "Created Interpreter");
    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return {nullptr, nullptr, nullptr, nullptr};
    }
    printf("Allocated tensors\n");
    ESP_LOGI("get_model", "Allocated tensors");

    int arena_used_bytes = interpreter->arena_used_bytes();

    printf("Created Interpreter\n Arena used bytes: %d\n", arena_used_bytes);
    
    TfLiteTensor* input = interpreter->input(0);
    TfLiteTensor* output = interpreter->output(0);
    
    return {model, interpreter, input, output};
}

extern "C" void app_main(void)
{
    printf("Started @ app_main\n");
    ESP_LOGI("app_main", "Started @ app_main");

    // initialize variables
    // const tflite::Model* palm_detection_model = nullptr;
    // tflite::MicroInterpreter* palm_detection_interpreter = nullptr;
    // TfLiteTensor* palm_detection_input = nullptr;
    // TfLiteTensor* palm_detection_output = nullptr;

    // const tflite::Model* hand_landmark_model = nullptr;
    // tflite::MicroInterpreter* hand_landmark_interpreter = nullptr;
    // TfLiteTensor* hand_landmark_input = nullptr;
    // TfLiteTensor* hand_landmark_output = nullptr;


    model_info palm_detection_model_info = get_model(palm_detection_full_tflite);
    // model_info hand_landmark_model_info = get_model(hand_landmark_lite_tflite);

    // [ palm_tracking_model, palm_detection_interpreter, palm_detection_input, palm_detection_output ] = palm_detection_model_info;
    // [ hand_landmark_model, hand_landmark_interpreter, hand_landmark_input, hand_landmark_output ] = hand_landmark_model_info;
    // 1. load palm detection model
    // debug  - create interpreter
    // 2. load hand tracking model
    
}