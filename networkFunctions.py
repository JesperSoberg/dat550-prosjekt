import torch


def finalPrediction(predictions):
    result = []
    for prediction in predictions:
        final_prediction = [0]*10
        index = prediction.argmax()
        final_prediction[index] = 1
        result.append(final_prediction)
    return torch.tensor(result)