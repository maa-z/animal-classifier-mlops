import torch


def predict_image(
    image,
    model,
    transform,
    class_mapping,
    device
):

    image_tensor = transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        device
    )

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_idx = (
            torch.max(
                probabilities,
                dim=1
            )
        )

    predicted_class = (
        class_mapping[
            str(
                predicted_idx.item()
            )
        ]
    )

    return {
        "prediction": predicted_class,
        "confidence": round(
            confidence.item() * 100,
            2
        )
    }