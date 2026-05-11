// ui.js - Visualización y manipulación de la interfaz
const UI = (function () {

    const CLASS_NAMES  = ['Diagonal ↘', 'Diagonal ↗', 'Horizontal ━', 'Vertical │'];
    const CLASS_COLORS = ['#e53935',    '#1e88e5',    '#43a047',      '#fb8c00'];

    const elements = {};

    // ─── Inicialización ───────────────────────────────────────────────────────

    function init() {
        elements.datasetGrid     = document.getElementById('dataset-grid');
        elements.batchHighlight  = document.getElementById('batch-highlight');
        elements.inputMap        = document.getElementById('input-map');
        elements.convMap         = document.getElementById('conv-map');
        elements.convInfoLabel   = document.getElementById('conv-info-label');
        elements.convKernels     = document.getElementById('conv-kernels');
        elements.poolMap         = document.getElementById('pool-map');
        elements.poolInfoLabel   = document.getElementById('pool-info-label');
        elements.poolWindows     = document.getElementById('pool-windows');
        elements.flattenValues   = document.getElementById('flatten-values');
        elements.dense1Values    = document.getElementById('dense1-values');
        elements.outputValues    = document.getElementById('output-values');
        elements.stepDesc        = document.getElementById('step-description');
        elements.lossValue       = document.getElementById('loss-value');
        elements.testAcc         = document.getElementById('test-acc');
        elements.currentEpoch    = document.getElementById('current-epoch');
        elements.lossPlot        = document.getElementById('loss-plot');
        elements.downloadSection = document.getElementById('download-section');
        elements.layersView      = document.getElementById('layers-view');
        elements.predictionsPanel = document.getElementById('predictions-panel');

        // Tooltip filtros convolucionales
        const convTip = document.getElementById('filter-tooltip');
        elements.convKernels.addEventListener('mouseenter', () => convTip.classList.add('visible'));
        elements.convKernels.addEventListener('mouseleave', () => convTip.classList.remove('visible'));
        elements.convKernels.addEventListener('mousemove',  e => _positionTooltip(convTip, e));

        // Tooltip MaxPooling — activo sobre el mapa de salida y las ventanas
        const poolTip = document.getElementById('pool-tooltip');
        const poolBox = document.getElementById('layer-pool');
        poolBox.addEventListener('mouseenter', () => poolTip.classList.add('visible'));
        poolBox.addEventListener('mouseleave', () => poolTip.classList.remove('visible'));
        poolBox.addEventListener('mousemove',  e => _positionTooltip(poolTip, e));

        document.getElementById('lr').addEventListener('input',
            e => document.getElementById('lr-value').textContent = e.target.value);
        document.getElementById('epochs').addEventListener('input',
            e => document.getElementById('epochs-value').textContent = e.target.value);
        document.getElementById('batch-size').addEventListener('input',
            e => document.getElementById('batch-size-value').textContent = e.target.value);
        document.getElementById('dropout').addEventListener('input',
            e => document.getElementById('dropout-value').textContent = e.target.value);
    }

    // ─── Dataset ──────────────────────────────────────────────────────────────

    // trainCount    : cuántas imágenes son de entrenamiento (las demás son test → violeta)
    // batchIndices  : imágenes del lote actual en entrenamiento → azul
    // activeIndices : ejemplo siendo procesado en entrenamiento → naranja
    // evalIndices   : ejemplos siendo evaluados en la fase test → verde
    function renderDataset(images, labels, batchIndices = [], activeIndices = [],
                           trainCount = images.length, evalIndices = []) {
        elements.datasetGrid.innerHTML = '';
        images.forEach((img, idx) => {
            const card = document.createElement('div');
            const isEval   = evalIndices.includes(idx);
            const isActive = activeIndices.includes(idx);
            const isBatch  = batchIndices.includes(idx);
            const isTest   = idx >= trainCount;
            let cls = 'image-card';
            if      (isEval)   cls += ' eval-active';
            else if (isActive) cls += ' active';
            else if (isBatch)  cls += ' batch';
            else if (isTest)   cls += ' test-set';
            card.className = cls;
            const grid = _makePixelGrid(img, 'dataset');
            const labelSpan = document.createElement('div');
            labelSpan.textContent = `C${labels[idx]}`;
            card.appendChild(grid);
            card.appendChild(labelSpan);
            elements.datasetGrid.appendChild(card);
        });
    }

    // ─── Limpieza de capas ────────────────────────────────────────────────────

    function clearLayerDisplays() {
        elements.inputMap.innerHTML      = '';
        elements.convMap.innerHTML       = '';
        elements.convInfoLabel.textContent = '';
        elements.convKernels.innerHTML   = '';
        elements.poolMap.innerHTML       = '';
        elements.poolInfoLabel.textContent = '';
        elements.poolWindows.innerHTML   = '';
        elements.flattenValues.innerHTML = '';
        elements.dense1Values.innerHTML  = '';
        elements.outputValues.innerHTML  = '';
    }

    // ─── Mapas de activación ──────────────────────────────────────────────────
    //
    // maps: [H][W][C]   (canales al final)
    // opts.scheme : 'binary' | 'diverging' | 'sequential'
    // opts.absMax : (opcional) escala externa para normalización
    //              Si se pasa, los colores de conv y pool serán comparables.

    function renderFeatureMaps(maps, container, opts = {}) {
        const scheme = opts.scheme || 'diverging';
        const H = maps.length, W = maps[0].length, C = maps[0][0].length;

        // Calcular min/max globales de este conjunto de mapas
        let vMin = Infinity, vMax = -Infinity;
        for (let i = 0; i < H; i++)
            for (let j = 0; j < W; j++)
                for (let c = 0; c < C; c++) {
                    const v = maps[i][j][c];
                    if (v < vMin) vMin = v;
                    if (v > vMax) vMax = v;
                }

        // Si se pasa escala externa, usarla (para conv ↔ pool consistente)
        const absMax = opts.absMax !== undefined
            ? opts.absMax
            : Math.max(Math.abs(vMin), Math.abs(vMax), 1e-6);

        for (let c = 0; c < C; c++) {
            const group = document.createElement('div');
            group.className = 'filter-group';

            // Etiqueta del filtro (solo para capas con más de 1 canal)
            if (C > 1) {
                const lbl = document.createElement('div');
                lbl.className = 'filter-label';
                lbl.textContent = `F${c + 1}`;
                group.appendChild(lbl);
            }

            const mapDiv = document.createElement('div');
            mapDiv.className = 'feature-map';
            mapDiv.style.gridTemplateColumns = `repeat(${W}, 10px)`;
            mapDiv.style.display = 'grid';

            for (let i = 0; i < H; i++) {
                for (let j = 0; j < W; j++) {
                    const val = maps[i][j][c];
                    const cell = document.createElement('div');
                    cell.className = 'map-pixel';
                    cell.title = val.toFixed(3);
                    cell.style.backgroundColor = _pickColor(val, absMax, scheme);
                    mapDiv.appendChild(cell);
                }
            }

            group.appendChild(mapDiv);
            container.appendChild(group);
        }
    }

    // ─── Colores ──────────────────────────────────────────────────────────────

    // Interpola linealmente entre dos colores RGB
    function _lerp(c1, c2, t) {
        const r = Math.round(c1[0] + (c2[0] - c1[0]) * t);
        const g = Math.round(c1[1] + (c2[1] - c1[1]) * t);
        const b = Math.round(c1[2] + (c2[2] - c1[2]) * t);
        return `rgb(${r},${g},${b})`;
    }

    // blue ──── white ──── red  (para conv pre-ReLU)
    const _BLUE = [39, 100, 176];
    const _WHITE = [255, 255, 255];
    const _RED   = [197, 27, 27];
    // blanco azulado ──── rojo  (para pool post-ReLU, misma escala roja que conv)
    const _NEAR_WHITE = [240, 244, 255];

    function _pickColor(val, absMax, scheme) {
        if (scheme === 'binary') {
            // 0 → blanco, 1 → gris oscuro
            return val > 0.5 ? 'rgb(44,62,80)' : 'rgb(255,255,255)';
        }
        if (scheme === 'diverging') {
            const t = Math.max(-1, Math.min(1, val / absMax));
            return t >= 0
                ? _lerp(_WHITE, _RED, t)
                : _lerp(_WHITE, _BLUE, -t);
        }
        // sequential: 0 (blanco azulado) → rojo (misma escala que diverging positiva)
        const t = Math.max(0, Math.min(1, val / absMax));
        return _lerp(_NEAR_WHITE, _RED, t);
    }

    // ─── Arrays 1D (flatten / dense / output) ────────────────────────────────

    function render1DArray(arr, container) {
        container.innerHTML = '';
        const p = document.createElement('p');
        p.textContent = arr.map(v => v.toFixed(3)).join(', ');
        container.appendChild(p);
    }

    // ─── Estadísticas ─────────────────────────────────────────────────────────

    function updateStepDescription(text) {
        elements.stepDesc.textContent = text;
    }

    function updateStats(loss, acc) {
        elements.lossValue.textContent = loss !== null ? loss.toFixed(4) : '---';
        elements.testAcc.textContent   = acc  !== null ? (acc * 100).toFixed(1) + '%' : '---';
    }

    function updateEpoch(epoch) {
        elements.currentEpoch.textContent = epoch;
    }

    function plotLoss(lossHistory) {
        const canvas = elements.lossPlot;
        canvas.innerHTML = '';
        const w = canvas.clientWidth, h = canvas.clientHeight;
        const svg = _svgEl('svg');
        svg.setAttribute('width', w);
        svg.setAttribute('height', h);
        if (lossHistory.length < 2) { canvas.appendChild(svg); return; }

        const minL = Math.min(...lossHistory);
        const maxL = Math.max(...lossHistory);
        const range = maxL - minL || 1;
        const pts = lossHistory.map((l, i) => ({
            x: (i / (lossHistory.length - 1)) * w,
            y: h - ((l - minL) / range) * (h - 20) - 10
        }));
        let d = `M${pts[0].x},${pts[0].y}`;
        for (let i = 1; i < pts.length; i++) d += ` L${pts[i].x},${pts[i].y}`;
        const path = _svgEl('path');
        path.setAttribute('d', d);
        path.setAttribute('stroke', '#3498db');
        path.setAttribute('stroke-width', '2');
        path.setAttribute('fill', 'none');
        svg.appendChild(path);
        canvas.appendChild(svg);
    }

    function _svgEl(tag) {
        return document.createElementNS('http://www.w3.org/2000/svg', tag);
    }

    // ─── Botón de descarga ────────────────────────────────────────────────────

    function showDownloadButton() {
        elements.downloadSection.style.display = 'block';
    }

    // ─── Parámetros ───────────────────────────────────────────────────────────

    function getHyperparams() {
        return {
            lr:           parseFloat(document.getElementById('lr').value),
            epochs:       parseInt(document.getElementById('epochs').value),
            batchSize:    parseInt(document.getElementById('batch-size').value),
            dropoutKeep:  parseFloat(document.getElementById('dropout').value),
            filters:      parseInt(document.getElementById('filters').value)
        };
    }

    // ─── Panel de predicciones ────────────────────────────────────────────────

    function showPredictionsPanel() {
        elements.layersView.style.display      = 'none';
        elements.predictionsPanel.style.display = 'flex';
    }

    function showLayersView() {
        elements.predictionsPanel.style.display = 'none';
        elements.layersView.style.display       = 'flex';
    }

    // samples: [{ img, trueLabel, probs }]
    function renderPredictions(samples) {
        const panel = elements.predictionsPanel;
        panel.innerHTML = '';

        // Título del panel
        const nCorrect = samples.filter(
            s => s.probs.indexOf(Math.max(...s.probs)) === s.trueLabel
        ).length;

        const title = document.createElement('div');
        title.className = 'pred-panel-title';
        title.innerHTML =
            `Predicciones de la red entrenada &nbsp;<small>` +
            `(${nCorrect} de ${samples.length} correctas en esta selección)</small>`;
        panel.appendChild(title);

        const row = document.createElement('div');
        row.className = 'pred-cards-row';

        samples.forEach((s, idx) => {
            const predIdx = s.probs.indexOf(Math.max(...s.probs));
            const correct = predIdx === s.trueLabel;
            row.appendChild(_buildPredCard(s, idx, predIdx, correct));
        });

        panel.appendChild(row);
    }

    function _buildPredCard(s, idx, predIdx, correct) {
        const card = document.createElement('div');
        card.className = `pred-card ${correct ? 'pred-correct' : 'pred-wrong'}`;

        // Número de muestra
        const num = document.createElement('div');
        num.className = 'pred-num';
        num.textContent = `Muestra ${idx + 1}`;
        card.appendChild(num);

        // Imagen 8×8
        const imgRow = document.createElement('div');
        imgRow.className = 'pred-image-row';
        imgRow.appendChild(_makePixelGrid(s.img, 'pred'));
        card.appendChild(imgRow);

        // Badge correcto / incorrecto
        const badge = document.createElement('div');
        badge.className = `pred-badge ${correct ? 'badge-correct' : 'badge-wrong'}`;
        badge.textContent = correct ? '✓ Correcto' : '✗ Incorrecto';
        card.appendChild(badge);

        // Etiquetas real y predicha
        const labels = document.createElement('div');
        labels.className = 'pred-labels';
        labels.innerHTML =
            `<div><span class="lbl-key">Real:&nbsp;</span>${CLASS_NAMES[s.trueLabel]}</div>` +
            `<div class="lbl-pred"><span class="lbl-key">Pred:&nbsp;</span>` +
            `<strong style="color:${CLASS_COLORS[predIdx]}">${CLASS_NAMES[predIdx]}</strong></div>`;
        card.appendChild(labels);

        // Separador
        const hr = document.createElement('hr');
        hr.className = 'pred-divider';
        card.appendChild(hr);

        // Barras de probabilidad
        const bars = document.createElement('div');
        bars.className = 'pred-bars';

        // Ordenar por probabilidad descendente para visualización
        const order = [0, 1, 2, 3].sort((a, b) => s.probs[b] - s.probs[a]);
        order.forEach(ci => {
            const pct = (s.probs[ci] * 100).toFixed(1);
            const row = document.createElement('div');
            row.className = `pred-bar-row${ci === predIdx ? ' bar-top' : ''}`;
            row.innerHTML =
                `<span class="bar-class-name">${CLASS_NAMES[ci]}</span>` +
                `<div class="bar-track">` +
                `  <div class="bar-fill" style="width:${pct}%;background:${CLASS_COLORS[ci]}"></div>` +
                `</div>` +
                `<span class="bar-pct">${pct}%</span>`;
            bars.appendChild(row);
        });
        card.appendChild(bars);

        return card;
    }

    // ─── Detalle capa convolucional ───────────────────────────────────────────

    // Etiqueta estática con la arquitectura de la capa conv
    function renderConvInfo(container, numFilters, kernelSize, inH, inW, outH, outW) {
        container.textContent =
            `${kernelSize}×${kernelSize} · stride 1 · padding válido` +
            `  |  ${inH}×${inW} → ${outH}×${outW} · ${numFilters} filtro${numFilters > 1 ? 's' : ''}`;
    }

    // Muestra todos los kernels 3×3 como grids coloreados + valor numérico
    // weights: [F][ci][kh][kw]  (ci = 1 para escala de grises)
    function renderConvKernels(weights, container) {
        container.innerHTML = '';
        weights.forEach((kernelSet, f) => {
            const kernel = kernelSet[0]; // canal único
            const kSize  = kernel.length;

            let absMax = 1e-6;
            kernel.forEach(row => row.forEach(v => {
                if (Math.abs(v) > absMax) absMax = Math.abs(v);
            }));

            const group = document.createElement('div');
            group.className = 'filter-group';

            const lbl = document.createElement('div');
            lbl.className = 'filter-label';
            lbl.textContent = `F${f + 1}`;
            group.appendChild(lbl);

            const grid = document.createElement('div');
            grid.className = 'kernel-grid';
            grid.style.gridTemplateColumns = `repeat(${kSize}, 17px)`;

            kernel.forEach(row => {
                row.forEach(val => {
                    const cell = document.createElement('div');
                    cell.className = 'kernel-cell';
                    cell.style.backgroundColor = _pickColor(val, absMax, 'diverging');
                    cell.textContent = val.toFixed(2);
                    // texto claro sobre fondos saturados, oscuro sobre blanco
                    cell.style.color = Math.abs(val) / absMax > 0.45
                        ? 'rgba(255,255,255,0.92)'
                        : 'rgba(0,0,0,0.72)';
                    grid.appendChild(cell);
                });
            });

            group.appendChild(grid);
            container.appendChild(group);
        });
    }

    // ─── Detalle capa de pooling ──────────────────────────────────────────────

    function renderPoolInfo(container, inH, inW, inC, outH, outW) {
        container.textContent =
            `${inH}×${inW}×${inC} → ${outH}×${outW}×${inC}` +
            `  |  ventana 2×2 · stride 2 · conserva el máximo de 4 valores`;
    }

    // Muestra la entrada al MaxPool (post-ReLU) con las ventanas 2×2 delimitadas
    // y el máximo de cada ventana resaltado con borde naranja.
    // reluOut : [H][W][C]   switches : [C][outH][outW] = [maxI, maxJ]
    function renderPoolWindows(reluOut, switches, container) {
        container.innerHTML = '';
        const H = reluOut.length, W = reluOut[0].length, C = reluOut[0][0].length;

        for (let c = 0; c < C; c++) {
            // Construir set de posiciones máximas para este canal
            const maxSet = new Set();
            switches[c].forEach(row => row.forEach(([mi, mj]) => maxSet.add(`${mi},${mj}`)));

            // absMax del canal para normalizar colores
            let absMax = 1e-6;
            for (let i = 0; i < H; i++)
                for (let j = 0; j < W; j++)
                    if (reluOut[i][j][c] > absMax) absMax = reluOut[i][j][c];

            const group = document.createElement('div');
            group.className = 'filter-group';

            const lbl = document.createElement('div');
            lbl.className = 'filter-label';
            lbl.textContent = `F${c + 1}`;
            group.appendChild(lbl);

            const grid = document.createElement('div');
            grid.className = 'pool-window-grid';
            grid.style.gridTemplateColumns = `repeat(${W}, 16px)`;

            for (let i = 0; i < H; i++) {
                for (let j = 0; j < W; j++) {
                    const val   = reluOut[i][j][c];
                    const isMax = maxSet.has(`${i},${j}`);

                    // Separadores entre ventanas 2×2
                    const rBorder = j % 2 === 1 && j < W - 1;
                    const bBorder = i % 2 === 1 && i < H - 1;
                    let cls = 'pool-window-cell';
                    if (isMax)              cls += ' is-max';
                    if (rBorder && bBorder) cls += ' win-border-rb';
                    else if (rBorder)       cls += ' win-border-r';
                    else if (bBorder)       cls += ' win-border-b';

                    const cell = document.createElement('div');
                    cell.className = cls;
                    cell.style.backgroundColor = _pickColor(val, absMax, 'sequential');
                    cell.textContent = val.toFixed(1);
                    cell.style.color = val / absMax > 0.45
                        ? 'rgba(255,255,255,0.92)'
                        : 'rgba(0,0,0,0.65)';
                    grid.appendChild(cell);
                }
            }

            group.appendChild(grid);
            container.appendChild(group);
        }
    }

    // ─── Helpers ──────────────────────────────────────────────────────────────

    // Posiciona el tooltip cerca del cursor sin salirse de la ventana
    function _positionTooltip(tooltip, e) {
        const gap = 14;
        const tw  = tooltip.offsetWidth  || 330;
        const th  = tooltip.offsetHeight || 220;
        let left  = e.clientX + gap;
        let top   = e.clientY + gap;
        if (left + tw > window.innerWidth  - 8) left = e.clientX - tw - gap;
        if (top  + th > window.innerHeight - 8) top  = e.clientY - th - gap;
        tooltip.style.left = left + 'px';
        tooltip.style.top  = top  + 'px';
    }

    // mode: 'dataset' | 'pred'
    function _makePixelGrid(img, mode) {
        const grid = document.createElement('div');
        grid.className = mode === 'pred' ? 'pred-pixel-grid' : 'pixel-grid';
        for (let i = 0; i < 8; i++) {
            for (let j = 0; j < 8; j++) {
                const pix = document.createElement('div');
                pix.style.backgroundColor = img[i][j] ? '#2c3e50' : '#ffffff';
                grid.appendChild(pix);
            }
        }
        return grid;
    }

    return {
        init,
        renderDataset,
        clearLayerDisplays,
        renderFeatureMaps,
        renderConvInfo,
        renderConvKernels,
        renderPoolInfo,
        renderPoolWindows,
        render1DArray,
        updateStepDescription,
        updateStats,
        updateEpoch,
        plotLoss,
        showDownloadButton,
        getHyperparams,
        showPredictionsPanel,
        showLayersView,
        renderPredictions,
        elements
    };
})();
