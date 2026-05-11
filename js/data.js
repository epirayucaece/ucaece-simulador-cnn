// data.js - Generación del dataset sintético de formas geométricas 8x8
const DataGenerator = (function() {
    // Clases: 0 = diagonal descendente, 1 = diagonal ascendente,
    //         2 = banda horizontal, 3 = banda vertical
    const N_CLASSES = 4;
    const IMG_SIZE = 8;

    // Función auxiliar: crea imagen base de una clase
    function basePattern(cls) {
        const img = Array(IMG_SIZE).fill().map(() => Array(IMG_SIZE).fill(0));
        if (cls === 0) { // diagonal principal
            for (let i = 0; i < IMG_SIZE; i++) img[i][i] = 1;
        } else if (cls === 1) { // anti-diagonal
            for (let i = 0; i < IMG_SIZE; i++) img[i][IMG_SIZE-1-i] = 1;
        } else if (cls === 2) { // banda horizontal (filas 3 y 4)
            for (let j = 0; j < IMG_SIZE; j++) {
                img[2][j] = 1;
                img[3][j] = 1;
            }
        } else if (cls === 3) { // banda vertical (columnas 3 y 4)
            for (let i = 0; i < IMG_SIZE; i++) {
                img[i][2] = 1;
                img[i][3] = 1;
            }
        }
        return img;
    }

    // Agrega ruido controlado (voltea algunos píxeles) conservando la clase
    function addNoise(img, noiseLevel = 0.05) {
        const noisy = img.map(row => [...row]);
        for (let i = 0; i < IMG_SIZE; i++) {
            for (let j = 0; j < IMG_SIZE; j++) {
                if (Math.random() < noiseLevel) {
                    noisy[i][j] = 1 - noisy[i][j];
                }
            }
        }
        return noisy;
    }

    // Genera un conjunto completo con variaciones
    function generateDataset(samplesPerClass = 10, noise = 0.05) {
        const images = [];
        const labels = [];
        for (let cls = 0; cls < N_CLASSES; cls++) {
            const base = basePattern(cls);
            for (let s = 0; s < samplesPerClass; s++) {
                const img = (s === 0) ? base : addNoise(base, noise);
                images.push(img);
                labels.push(cls);
            }
        }
        return { images, labels };
    }

    // Mezcla aleatoria manteniendo pares (imagen, etiqueta)
    function shuffleTogether(images, labels) {
        const indices = [...Array(images.length).keys()];
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        const shufImages = indices.map(i => images[i]);
        const shufLabels = indices.map(i => labels[i]);
        return { images: shufImages, labels: shufLabels };
    }

    // Divide en entrenamiento y prueba (80/20)
    function trainTestSplit(images, labels, trainRatio = 0.8) {
        const n = images.length;
        const nTrain = Math.floor(n * trainRatio);
        const trainImgs = images.slice(0, nTrain);
        const trainLbls = labels.slice(0, nTrain);
        const testImgs = images.slice(nTrain);
        const testLbls = labels.slice(nTrain);
        return { trainImgs, trainLbls, testImgs, testLbls };
    }

    // Crea lotes para entrenamiento
    function createBatches(images, labels, batchSize) {
        const batches = [];
        for (let i = 0; i < images.length; i += batchSize) {
            batches.push({
                imgs: images.slice(i, i + batchSize),
                lbls: labels.slice(i, i + batchSize)
            });
        }
        return batches;
    }

    return {
        generateDataset,
        shuffleTogether,
        trainTestSplit,
        createBatches,
        IMG_SIZE,
        N_CLASSES
    };
})();