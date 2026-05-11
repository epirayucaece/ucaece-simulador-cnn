// main.js - Orquestador de la simulación (máquina de estados)
(function() {
    // Referencias a módulos globales
    const dataGen = DataGenerator;
    const net = Network;
    const opt = Optimizer;

    // Estado global de la simulación
    let dataset, trainImgs, trainLbls, testImgs, testLbls;
    let batches, currentBatchIdx = 0;
    let currentEpoch = 0, totalEpochs;
    let lossHistory = [];
    let model, optimizer;
    let phase = 'init'; // init, dataset_loaded, training, testing, done
    let stepIdx = 0;     // sub-paso dentro de una fase
    let selectedSampleIdx = null; // imagen en foco
    let currentBatchSampleIdx = 0;

    // Botones
    const btnStep = document.getElementById('btn-step');
    const btnEpoch = document.getElementById('btn-epoch');
    const btnRun = document.getElementById('btn-run');
    const btnReset = document.getElementById('btn-reset');
    const btnDownload = document.getElementById('btn-download');

    UI.init();
    resetSimulation();

    btnStep.addEventListener('click', executeStep);
    btnEpoch.addEventListener('click', executeOneEpoch);
    btnRun.addEventListener('click', executeAll);
    btnReset.addEventListener('click', resetSimulation);
    btnDownload.addEventListener('click', downloadExplanation);

    function resetSimulation() {
        phase = 'init';
        stepIdx = 0;
        currentEpoch = 0;
        lossHistory = [];
        UI.clearLayerDisplays();
        UI.showLayersView();
        UI.updateStepDescription('Preparando dataset... Presiona "Siguiente paso".');
        UI.updateStats(null, null);
        UI.updateEpoch(0);
        UI.plotLoss([]);
        document.getElementById('download-section').style.display = 'none';
        selectedSampleIdx = null;
        // Generar dataset
        const raw = dataGen.generateDataset(10, 0.05);
        const shuffled = dataGen.shuffleTogether(raw.images, raw.labels);
        const split = dataGen.trainTestSplit(shuffled.images, shuffled.labels, 0.8);
        dataset = split;
        trainImgs = split.trainImgs;
        trainLbls = split.trainLbls;
        testImgs = split.testImgs;
        testLbls = split.testLbls;
        const hp = UI.getHyperparams();
        batches = dataGen.createBatches(trainImgs, trainLbls, hp.batchSize);
        currentBatchIdx = 0;
        totalEpochs = hp.epochs;
        model = new net.CNNModel({ filters: hp.filters, dropoutKeep: hp.dropoutKeep });
        optimizer = new opt.Adam(hp.lr);
        UI.renderDataset([...trainImgs, ...testImgs], [...trainLbls, ...testLbls], [], [], trainImgs.length);
        ArchDiagram.render(hp.filters);
        ArchDiagram.clearHighlight();
        phase = 'dataset_loaded';
    }

    function executeStep() {
        if (phase === 'init' || phase === 'done') return;
        const hp = UI.getHyperparams();
        if (phase === 'dataset_loaded') {
            // Mostrar primer lote y pasar a entrenamiento
            showCurrentBatch();
            phase = 'training';
            stepIdx = 0;
            currentBatchSampleIdx = 0;
            UI.updateStepDescription('Iniciando entrenamiento. Primer lote mostrado.');
        } else if (phase === 'training') {
            // Sub-pasos dentro del procesamiento de un ejemplo
            trainingStep();
        } else if (phase === 'testing') {
            performTestingStep();
        }
    }

    function showCurrentBatch() {
        const batch = batches[currentBatchIdx];
        const batchIndices = batch.imgs.map(img => trainImgs.indexOf(img));
        UI.renderDataset([...trainImgs, ...testImgs], [...trainLbls, ...testLbls], batchIndices, [], trainImgs.length);
        UI.updateStepDescription(`Lote ${currentBatchIdx+1}/${batches.length}: ${batch.imgs.length} imágenes.`);
    }

    function trainingStep() {
        if (currentEpoch >= totalEpochs) {
            phase = 'testing';
            UI.updateStepDescription('Entrenamiento completado. Evaluando en conjunto de prueba.');
            return;
        }
        const hp = UI.getHyperparams();
        const batch = batches[currentBatchIdx];
        if (currentBatchSampleIdx >= batch.imgs.length) {
            // Fin del lote, actualizar parámetros
            finishBatch();
            currentBatchIdx++;
            if (currentBatchIdx >= batches.length) {
                currentBatchIdx = 0;
                currentEpoch++;
                UI.updateEpoch(currentEpoch);
                // Evaluar pérdida en test
                const testLoss = computeTestLoss();
                const testAcc = computeTestAccuracy();
                UI.updateStats(testLoss, testAcc);
                lossHistory.push(testLoss);
                UI.plotLoss(lossHistory);
                if (currentEpoch >= totalEpochs) {
                    phase = 'testing';
                    UI.updateStepDescription('Entrenamiento completado. Pasando a prueba detallada.');
                    showDownloadButton();
                    return;
                }
            }
            showCurrentBatch();
            currentBatchSampleIdx = 0;
            return; // el siguiente paso procesará el primer ejemplo del nuevo lote
        }

        const img = batch.imgs[currentBatchSampleIdx];
        const label = batch.lbls[currentBatchSampleIdx];
        selectedSampleIdx = trainImgs.indexOf(img);

        // Resaltar el lote completo (azul) y el ejemplo actual (naranja)
        const batchIndices = batches[currentBatchIdx].imgs.map(i => trainImgs.indexOf(i));
        UI.renderDataset([...trainImgs, ...testImgs], [...trainLbls, ...testLbls], batchIndices, [selectedSampleIdx], trainImgs.length);

        // Forward pass
        const out = model.forward(img, true);
        const yTrue = Array(4).fill(0);
        yTrue[label] = 1;
        const dout = out.map((o,i) => o - yTrue[i]); // gradiente softmax+crossentropy

        // Backward inmediato: los cachés de activaciones corresponden a este ejemplo.
        // Guardamos el mapa de gradientes de pesos (no solo dout) para promediar correctamente.
        const fullGrads = model.backward(dout, optimizer.lr);
        const gradMap = flattenGradients(fullGrads, model);
        if (!batch.accumulatedGrads) batch.accumulatedGrads = [];
        batch.accumulatedGrads.push(gradMap);

        // Visualización DESPUÉS del backward para no sobreescribir los cachés de activación
        visualizeSample(img, out, label);
        UI.updateStepDescription(`Ejemplo ${currentBatchSampleIdx+1}/${batch.imgs.length} — lote ${currentBatchIdx+1}. Real: ${label}, Pred: ${out.indexOf(Math.max(...out))}`);
        currentBatchSampleIdx++;
    }

    function finishBatch() {
        const batch = batches[currentBatchIdx];
        const allGradMaps = batch.accumulatedGrads;
        if (!allGradMaps || allGradMaps.length === 0) return;

        ArchDiagram.clearHighlight();
        // Promedio correcto: se promedian los gradientes de PESOS de cada ejemplo del lote
        const n = allGradMaps.length;
        const avgMap = {};
        for (const key in allGradMaps[0]) {
            avgMap[key] = averageNestedArrays(allGradMaps.map(m => m[key]), n);
        }
        const updates = optimizer.step(avgMap);
        applyUpdates(model, updates);
        batch.accumulatedGrads = [];
    }

    // Promedia recursivamente arrays anidados de cualquier profundidad
    function averageNestedArrays(arrays, n) {
        if (typeof arrays[0] === 'number') {
            return arrays.reduce((s, v) => s + v, 0) / n;
        }
        return arrays[0].map((_, i) => averageNestedArrays(arrays.map(a => a[i]), n));
    }

    function flattenGradients(gradObj, model) {
        const map = {};
        // conv weights 4D: [F][ci][kh][kw]
        const convW = gradObj.conv.dW;
        for (let f = 0; f < convW.length; f++) {
            for (let ci = 0; ci < convW[f].length; ci++) {
                map[`conv_W_${f}_${ci}`] = convW[f][ci];
            }
        }
        map['conv_b'] = gradObj.conv.db;
        map['dense1_W'] = gradObj.dense1.dW;
        map['dense1_b'] = gradObj.dense1.db;
        map['dense2_W'] = gradObj.dense2.dW;
        map['dense2_b'] = gradObj.dense2.db;
        return map;
    }

    function applyUpdates(model, updates) {
        // actualizar pesos de conv
        for (const key in updates) {
            const update = updates[key];
            if (key.startsWith('conv_W_')) {
                const parts = key.split('_');
                const f = parseInt(parts[2]), ci = parseInt(parts[3]);
                const weight = model.conv.weights[f][ci];
                for (let i = 0; i < weight.length; i++)
                    for (let j = 0; j < weight[i].length; j++)
                        weight[i][j] += update[i][j];
            } else if (key === 'conv_b') {
                for (let i = 0; i < model.conv.bias.length; i++)
                    model.conv.bias[i] += update[i];
            } else if (key === 'dense1_W') {
                for (let i = 0; i < model.dense1.weights.length; i++)
                    for (let j = 0; j < model.dense1.weights[i].length; j++)
                        model.dense1.weights[i][j] += update[i][j];
            } else if (key === 'dense1_b') {
                for (let i = 0; i < model.dense1.bias.length; i++)
                    model.dense1.bias[i] += update[i];
            } else if (key === 'dense2_W') {
                for (let i = 0; i < model.dense2.weights.length; i++)
                    for (let j = 0; j < model.dense2.weights[i].length; j++)
                        model.dense2.weights[i][j] += update[i][j];
            } else if (key === 'dense2_b') {
                for (let i = 0; i < model.dense2.bias.length; i++)
                    model.dense2.bias[i] += update[i];
            }
        }
    }

    function visualizeSample(img, outputProbs, trueLabel) {
        UI.clearLayerDisplays();

        // Entrada — [H][W][1], esquema binario (blanco/negro)
        ArchDiagram.highlight('input');
        const inputHWC = img.map(row => row.map(val => [val]));
        UI.renderFeatureMaps(inputHWC, document.getElementById('input-map'),
            { scheme: 'binary' });

        // Convolución (PRE-ReLU) — colormap divergente azul↔rojo
        // Rojo = respuesta positiva al filtro, Azul = respuesta negativa
        ArchDiagram.highlight('conv');
        const inp = [img];
        const convOut = model.conv.forward(inp);
        const convAbsMax = _absMax3D(convOut);
        UI.renderFeatureMaps(convOut, document.getElementById('conv-map'),
            { scheme: 'diverging', absMax: convAbsMax });

        // MaxPooling (POST-ReLU) — mismo absMax que conv para comparación directa
        // Blanco = activación eliminada por ReLU (era negativa), Rojo = activación retenida
        ArchDiagram.highlight('pool');
        const reluOut = model.relu.forward(convOut);
        const poolOut = model.pool.forward(reluOut);
        UI.renderFeatureMaps(poolOut, document.getElementById('pool-map'),
            { scheme: 'sequential', absMax: convAbsMax });

        // Flatten
        ArchDiagram.highlight('flatten');
        const flat = model.flatten.forward(poolOut);
        UI.render1DArray(flat, document.getElementById('flatten-values'));

        // Densa + Dropout
        ArchDiagram.highlight('dense1');
        const dense1Out = model.dense1.forward(flat);
        UI.render1DArray(dense1Out, document.getElementById('dense1-values'));

        // Softmax — usar el resultado real del forward de entrenamiento (con dropout)
        // para que la probabilidad mostrada sea coherente con la predicción reportada
        ArchDiagram.highlight('output');
        UI.render1DArray(outputProbs, document.getElementById('output-values'));
    }

    // Valor absoluto máximo de un tensor [H][W][C]
    function _absMax3D(maps) {
        let m = 1e-6;
        maps.forEach(row => row.forEach(px => px.forEach(v => {
            if (Math.abs(v) > m) m = Math.abs(v);
        })));
        return m;
    }

    function computeTestLoss() {
        const hp = UI.getHyperparams();
        let totalLoss = 0;
        for (let i = 0; i < testImgs.length; i++) {
            const out = model.forward(testImgs[i], false);
            const yTrue = Array(4).fill(0);
            yTrue[testLbls[i]] = 1;
            totalLoss += -Math.log(Math.max(out[testLbls[i]], 1e-7));
        }
        return totalLoss / testImgs.length;
    }

    function computeTestAccuracy() {
        let correct = 0;
        for (let i = 0; i < testImgs.length; i++) {
            const out = model.forward(testImgs[i], false);
            if (out.indexOf(Math.max(...out)) === testLbls[i]) correct++;
        }
        return correct / testImgs.length;
    }

    function performTestingStep() {
        // Seleccionar 4 muestras representativas (una por clase si es posible)
        const samples = selectPredictionSamples(4);

        // Inferencia con la red entrenada (sin dropout)
        const results = samples.map(({ img, label }) => ({
            img,
            trueLabel: label,
            probs: model.forward(img, false)
        }));

        // Resaltar en el grid las muestras evaluadas (verde)
        const evalIndices = samples.map(({ img }) => {
            const ti = testImgs.indexOf(img);
            return ti !== -1 ? trainImgs.length + ti : trainImgs.indexOf(img);
        });
        UI.renderDataset([...trainImgs, ...testImgs], [...trainLbls, ...testLbls],
            [], [], trainImgs.length, evalIndices);

        // Mostrar panel de predicciones en lugar de las capas
        UI.showPredictionsPanel();
        UI.renderPredictions(results);
        ArchDiagram.clearHighlight();

        const nCorrect = results.filter(
            r => r.probs.indexOf(Math.max(...r.probs)) === r.trueLabel
        ).length;
        const acc = computeTestAccuracy();
        UI.updateStepDescription(
            `Evaluación final — ${nCorrect}/${results.length} correctas en la selección. ` +
            `Precisión total en prueba: ${(acc * 100).toFixed(1)}%`
        );

        phase = 'done';
        showDownloadButton();
    }

    // Devuelve una muestra por cada clase (preferencia: conjunto de prueba)
    function selectPredictionSamples(n) {
        const byClass = [null, null, null, null];

        // Primero buscar en el conjunto de prueba
        for (let i = 0; i < testImgs.length; i++) {
            const c = testLbls[i];
            if (byClass[c] === null) byClass[c] = { img: testImgs[i], label: c };
        }
        // Completar clases faltantes con el conjunto de entrenamiento
        for (let i = 0; i < trainImgs.length; i++) {
            const c = trainLbls[i];
            if (byClass[c] === null) byClass[c] = { img: trainImgs[i], label: c };
        }

        return byClass.filter(Boolean).slice(0, n);
    }

    function executeOneEpoch() {
        // Avanza una época completa
        while (currentEpoch < totalEpochs) {
            for (currentBatchIdx = 0; currentBatchIdx < batches.length; currentBatchIdx++) {
                currentBatchSampleIdx = 0;
                while (currentBatchSampleIdx < batches[currentBatchIdx].imgs.length) {
                    trainingStep();
                }
                finishBatch();
            }
            currentBatchIdx = 0;
            currentEpoch++;
            UI.updateEpoch(currentEpoch);
            const testLoss = computeTestLoss();
            const testAcc = computeTestAccuracy();
            UI.updateStats(testLoss, testAcc);
            lossHistory.push(testLoss);
            UI.plotLoss(lossHistory);
            if (currentEpoch >= totalEpochs) break;
        }
        phase = 'testing';
        UI.updateStepDescription('Entrenamiento completado. Puede evaluar paso a paso.');
        showDownloadButton();
    }

    function executeAll() {
        executeOneEpoch();
        if (phase === 'testing') performTestingStep();
    }

    function showDownloadButton() {
        UI.showDownloadButton();
    }

    function downloadExplanation() {
        const hp = UI.getHyperparams();
        const finalAcc = computeTestAccuracy();
        const content = `
Simulación de Red Neuronal Convolucional (CNN)
===============================================
Cátedra de Sistemas Inteligentes

Parámetros utilizados:
- Tasa de aprendizaje: ${hp.lr}
- Épocas: ${hp.epochs}
- Tamaño de lote: ${hp.batchSize}
- Dropout (keep prob): ${hp.dropoutKeep}
- Filtros en capa convolucional: ${hp.filters}

Arquitectura de la red:
1. Entrada: imagen 8x8 en escala de grises
2. Convolución: ${hp.filters} filtros de 3x3, stride 1, padding válido
3. Activación ReLU
4. MaxPooling 2x2, stride 2
5. Aplanado (Flatten)
6. Capa densa oculta: 10 neuronas, ReLU
7. Dropout con probabilidad de retención ${hp.dropoutKeep}
8. Capa de salida: 4 neuronas, Softmax

Optimizador: Adam
Función de pérdida: Entropía cruzada categórica

Dataset sintético generado con 4 clases de formas geométricas:
- Diagonal descendente (/)
- Diagonal ascendente (\\)
- Banda horizontal
- Banda vertical
Total: 40 imágenes (80% entrenamiento, 20% prueba)

Resultado final:
Precisión en prueba: ${(finalAcc*100).toFixed(1)}%

Conceptos demostrados:
- Convolución y filtros aprendibles
- No linealidad ReLU
- Reducción de dimensionalidad con Pooling
- Regularización por Dropout
- Clasificación multiclase con Softmax
- Retropropagación y optimización con Adam

Esta simulación paso a paso permite observar cada capa y su función dentro de una CNN real.
        `;
        const blob = new Blob([content], {type: 'text/plain;charset=utf-8'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'explicacion_cnn.txt';
        a.click();
        URL.revokeObjectURL(url);
    }
})();