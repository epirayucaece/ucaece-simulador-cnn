// optimizer.js - Optimizador Adam
const Optimizer = (function() {
    class Adam {
        constructor(lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
            this.lr = lr;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.eps = eps;
            this.m = {}; // primer momento
            this.v = {}; // segundo momento
            this.t = 0;  // paso temporal
        }

        // Inicializa acumuladores para un conjunto de pesos
        initParam(key, shape) {
            if (!this.m[key]) {
                this.m[key] = zerosLike(shape);
                this.v[key] = zerosLike(shape);
            }
        }

        step(gradsMap) {
            this.t += 1;
            const updated = {};
            for (const key in gradsMap) {
                const grad = gradsMap[key];
                this.initParam(key, grad);
                const m = this.m[key];
                const v = this.v[key];

                // Actualizar momentos
                for (let i = 0; i < m.length; i++) {
                    if (Array.isArray(m[i])) {
                        for (let j = 0; j < m[i].length; j++) {
                            if (Array.isArray(m[i][j])) {
                                for (let k = 0; k < m[i][j].length; k++) {
                                    const g = grad[i][j][k];
                                    m[i][j][k] = this.beta1 * m[i][j][k] + (1 - this.beta1) * g;
                                    v[i][j][k] = this.beta2 * v[i][j][k] + (1 - this.beta2) * g * g;
                                }
                            } else {
                                m[i][j] = this.beta1 * m[i][j] + (1 - this.beta1) * grad[i][j];
                                v[i][j] = this.beta2 * v[i][j] + (1 - this.beta2) * grad[i][j] * grad[i][j];
                            }
                        }
                    } else {
                        m[i] = this.beta1 * m[i] + (1 - this.beta1) * grad[i];
                        v[i] = this.beta2 * v[i] + (1 - this.beta2) * grad[i] * grad[i];
                    }
                }

                // Corrección de sesgo y actualización de parámetros
                const mHat = scaleArray(m, 1 / (1 - Math.pow(this.beta1, this.t)));
                const vHat = scaleArray(v, 1 / (1 - Math.pow(this.beta2, this.t)));
                const paramUpdate = elementWiseOp(mHat, vHat, (mh, vh) => - this.lr * mh / (Math.sqrt(vh) + this.eps));
                updated[key] = paramUpdate;
            }
            return updated;
        }
    }

    // Funciones utilitarias para arrays multidimensionales
    function zerosLike(arr) {
        if (typeof arr === 'number') return 0;
        if (Array.isArray(arr)) {
            return arr.map(zerosLike);
        }
        return 0;
    }

    function scaleArray(arr, scalar) {
        return arr.map(x => Array.isArray(x) ? scaleArray(x, scalar) : x * scalar);
    }

    function elementWiseOp(arr1, arr2, op) {
        return arr1.map((x, i) =>
            Array.isArray(x) ? elementWiseOp(x, arr2[i], op) : op(x, arr2[i])
        );
    }

    return { Adam };
})();