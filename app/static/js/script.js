// hier sind die Layer, mit denen wir arbeiten können
const layers = {
    "Conv2D": {
        "filters": 32,
        "kernel_size": [2, 2],
        "strides": [4, 1],
    },
    "Dense": {
        "units": 32,
    },

    "AvgPooling2D": {},

    "Dropout": {
        "rate": 0.5
    },

    "Flatten": {},
    "BatchNormalization": {},
    "ReLU": {},
};

const textarea = document.getElementById("arch-json");

function addLayer(layerType) {
    let jsonToAdd = {};

    jsonToAdd["type"] = layerType;

    if (Object.keys(layers[layerType]).length > 0) {
        jsonToAdd["params"] = {};

        // get the params for the layer
        const params = layers[layerType];

        // add the params to the jsonToAdd
        for (const [key, value] of Object.entries(params)) {
            jsonToAdd["params"][key] = value;
        }
    }

    // finally, add json to the textarea
    textarea.value += (
        textarea.value.endsWith("\n") || textarea.value.length == 0 ? "" : "\n"
    ) + `${JSON.stringify(jsonToAdd)}\n`;
}
