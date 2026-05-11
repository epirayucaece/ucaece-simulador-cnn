// arch.js — Diagrama estático de arquitectura CNN estilo paper académico
const ArchDiagram = (function () {

    const PALETTE = {
        input:    '#546e7a',
        conv:     '#1565c0',
        pool:     '#2e7d32',
        flatten:  '#6a1b9a',
        dense1:   '#e65100',
        output:   '#b71c1c',
        arrow:    '#90a4ae',
        text:     '#263238',
        dim:      '#78909c',
        highlight:'#f9a825',
        bg:       '#fafbfc'
    };

    const CLASS_COLORS = ['#e53935','#1e88e5','#43a047','#fb8c00'];
    const CLASS_LABELS = ['Diag ↘','Diag ↗','Horiz','Vert'];

    let _svg = null;
    let _activeId = null;

    // ─── API pública ──────────────────────────────────────────────────────────

    function render(filters) {
        const container = document.getElementById('arch-diagram');
        if (!container) return;
        container.innerHTML = '';

        const VW = 810, VH = 124;
        const CY = 52;   // centro vertical de las capas
        const LY = VH - 14; // y de etiquetas inferiores (nombre)
        const SY = VH - 4;  // y de sublabels (dimensiones)

        const svg = _makeSVG(VW, VH);
        _svg = svg;

        // ── Definición de capas ────────────────────────────────────────────
        const layers = [
            {
                id: 'input', cx: 58, cy: CY,
                type: 'cube', w: 38, h: 74, depth: 9, count: 1,
                color: PALETTE.input,
                label: 'Entrada', sub: '8 × 8 × 1'
            },
            {
                id: 'conv', cx: 192, cy: CY,
                type: 'cube', w: 28, h: 58, depth: 8, count: filters,
                color: PALETTE.conv,
                label: 'Conv 3×3 + ReLU', sub: `6 × 6 × ${filters}`
            },
            {
                id: 'pool', cx: 320, cy: CY,
                type: 'cube', w: 20, h: 42, depth: 7, count: filters,
                color: PALETTE.pool,
                label: 'MaxPool 2×2', sub: `3 × 3 × ${filters}`
            },
            {
                id: 'flatten', cx: 420, cy: CY,
                type: 'cube', w: 12, h: 80, depth: 6, count: 1,
                color: PALETTE.flatten,
                label: 'Flatten', sub: `${9 * filters}`
            },
            {
                id: 'dense1', cx: 515, cy: CY,
                type: 'dots', dotR: 3.5, count: 10,
                color: PALETTE.dense1,
                label: 'Densa + Dropout', sub: '10 u.'
            },
            {
                id: 'output', cx: 650, cy: CY,
                type: 'output', dotR: 6.5, count: 4,
                color: PALETTE.output,
                label: 'Softmax', sub: '4 clases'
            }
        ];

        // ── Fondo y título del diagrama ───────────────────────────────────
        const bgRect = _el('rect');
        _attr(bgRect, { x:0, y:0, width: VW, height: VH,
                        fill: PALETTE.bg, rx:0 });
        svg.appendChild(bgRect);

        const title = _text('Arquitectura de la red CNN', 6, 9,
            { fill: PALETTE.dim, 'font-size': 7.5, 'font-style': 'italic',
              'text-anchor': 'start' });
        svg.appendChild(title);

        // ── Flechas (se dibujan antes para quedar detrás) ─────────────────
        const arrows = [
            { x1:104, x2:152, y: CY, label: 'convolución\n3×3 válida' },
            { x1:246, x2:289, y: CY, label: 'max pooling\n2×2' },
            { x1:356, x2:402, y: CY, label: 'aplanar' },
            { x1:434, x2:492, y: CY, label: 'FC' },
            { x1:528, x2:622, y: CY, label: 'FC + softmax' },
        ];
        arrows.forEach(a => _drawArrow(svg, a.x1, a.x2, a.y, a.label));

        // ── Capas ─────────────────────────────────────────────────────────
        layers.forEach(layer => {
            const g = _el('g');
            g.setAttribute('id', `arch-${layer.id}`);
            g.setAttribute('class', 'arch-layer');
            g.style.transition = 'opacity 0.3s, filter 0.3s';

            const stackOX = layer.count > 1 ? 4 : 0;
            const stackOY = layer.count > 1 ? -3 : 0;
            // Centro visual corregido por el offset del stack
            const visualCX = layer.cx + (layer.count - 1) * stackOX * 0.5;

            if (layer.type === 'cube') {
                // Dibujar cubos apilados (atrás hacia adelante)
                for (let i = layer.count - 1; i >= 0; i--) {
                    const ox = i * stackOX;
                    const oy = i * stackOY;
                    const alpha = Math.max(0.55, 1 - i * 0.12);
                    _drawCube(g,
                        layer.cx - layer.w / 2 + ox,
                        layer.cy - layer.h / 2 + oy,
                        layer.w, layer.h, layer.depth,
                        layer.color, alpha);
                }
            } else if (layer.type === 'dots') {
                _drawDots(g, layer.cx, layer.cy, layer.count, layer.dotR, layer.color);
            } else if (layer.type === 'output') {
                _drawOutputNodes(g, layer.cx, layer.cy, layer.count, layer.dotR);
            }

            svg.appendChild(g);

            // Etiqueta de nombre (negrita)
            const lx = layer.type === 'cube' ? visualCX : layer.cx;
            svg.appendChild(_text(layer.label, lx, LY,
                { fill: PALETTE.text, 'font-size': 8, 'font-weight': 'bold',
                  'text-anchor': 'middle' }));
            // Dimensiones (gris)
            svg.appendChild(_text(layer.sub, lx, SY,
                { fill: PALETTE.dim, 'font-size': 7.2, 'text-anchor': 'middle' }));
        });

        container.appendChild(svg);

        // Restaurar estado de highlight si existía
        if (_activeId) highlight(_activeId);
        else clearHighlight();
    }

    function highlight(layerId) {
        _activeId = layerId;
        if (!_svg) return;
        _svg.querySelectorAll('.arch-layer').forEach(g => {
            g.style.opacity = '0.25';
            g.style.filter  = '';
        });
        const target = _svg.querySelector(`#arch-${layerId}`);
        if (target) {
            target.style.opacity = '1';
            target.style.filter  = `drop-shadow(0 0 5px ${PALETTE.highlight})`;
        }
    }

    function clearHighlight() {
        _activeId = null;
        if (!_svg) return;
        _svg.querySelectorAll('.arch-layer').forEach(g => {
            g.style.opacity = '1';
            g.style.filter  = '';
        });
    }

    // ─── Helpers de dibujo ────────────────────────────────────────────────────

    function _drawCube(parent, x, y, w, h, depth, color, alpha) {
        const dx = depth * 0.78, dy = depth * 0.44;

        // Cara derecha
        _polygon(parent, [
            [x+w, y], [x+w+dx, y-dy], [x+w+dx, y+h-dy], [x+w, y+h]
        ], color, alpha * 0.5);

        // Cara superior
        _polygon(parent, [
            [x, y], [x+dx, y-dy], [x+w+dx, y-dy], [x+w, y]
        ], color, alpha * 0.72);

        // Cara frontal
        _polygon(parent, [
            [x, y], [x+w, y], [x+w, y+h], [x, y+h]
        ], color, alpha);
    }

    function _drawArrow(parent, x1, x2, y, label) {
        // Línea
        const line = _el('line');
        _attr(line, { x1, y1: y, x2: x2-6, y2: y,
                      stroke: PALETTE.arrow, 'stroke-width': 1.4,
                      'stroke-dasharray': '3,2' });
        parent.appendChild(line);

        // Punta
        _polygon(parent, [
            [x2, y], [x2-7, y-3], [x2-7, y+3]
        ], PALETTE.arrow, 1);

        // Etiqueta de la flecha (hasta 2 líneas)
        const mx = (x1 + x2) / 2;
        const lines = label.split('\n');
        lines.forEach((ln, i) => {
            parent.appendChild(_text(ln, mx, y - 18 + i * 9,
                { fill: PALETTE.dim, 'font-size': 6.8, 'text-anchor': 'middle',
                  'font-style': 'italic' }));
        });
    }

    function _drawDots(parent, cx, cy, count, r, color) {
        const gap = 3.5;
        const totalH = count * (r * 2) + (count - 1) * gap;
        for (let i = 0; i < count; i++) {
            const ny = cy - totalH / 2 + i * (r * 2 + gap) + r;
            const c = _el('circle');
            _attr(c, { cx, cy: ny, r,
                       fill: color, 'fill-opacity': 0.85,
                       stroke: color, 'stroke-width': 0.8 });
            parent.appendChild(c);
        }
    }

    function _drawOutputNodes(parent, cx, cy, count, r) {
        const gap = 5;
        const totalH = count * (r * 2) + (count - 1) * gap;
        for (let i = 0; i < count; i++) {
            const ny = cy - totalH / 2 + i * (r * 2 + gap) + r;
            const c = _el('circle');
            _attr(c, { cx, cy: ny, r,
                       fill: CLASS_COLORS[i], 'fill-opacity': 0.88,
                       stroke: CLASS_COLORS[i], 'stroke-width': 1 });
            parent.appendChild(c);
            parent.appendChild(_text(CLASS_LABELS[i], cx + r + 4, ny + 2.5,
                { fill: '#37474f', 'font-size': 6.5, 'text-anchor': 'start' }));
        }
    }

    function _polygon(parent, pts, fill, opacity) {
        const poly = _el('polygon');
        poly.setAttribute('points', pts.map(p => p.join(',')).join(' '));
        poly.setAttribute('fill', fill);
        poly.setAttribute('fill-opacity', opacity);
        poly.setAttribute('stroke', fill);
        poly.setAttribute('stroke-width', '0.5');
        poly.setAttribute('stroke-opacity', '0.6');
        parent.appendChild(poly);
    }

    function _text(content, x, y, attrs) {
        const t = _el('text');
        t.setAttribute('x', x);
        t.setAttribute('y', y);
        t.setAttribute('font-family', "Segoe UI, Arial, sans-serif");
        Object.entries(attrs).forEach(([k, v]) => t.setAttribute(k, v));
        t.textContent = content;
        return t;
    }

    function _el(tag) {
        return document.createElementNS('http://www.w3.org/2000/svg', tag);
    }

    function _attr(el, attrs) {
        Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
    }

    function _makeSVG(w, h) {
        const svg = _el('svg');
        _attr(svg, {
            viewBox: `0 0 ${w} ${h}`,
            width: '100%',
            height: '100%',
            preserveAspectRatio: 'xMidYMid meet'
        });
        return svg;
    }

    return { render, highlight, clearHighlight };
})();
