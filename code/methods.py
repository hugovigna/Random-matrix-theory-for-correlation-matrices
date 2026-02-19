"""methods.py
 Implementations of covariance cleaning routines.

Provides:
- class `cleaning_methods(X)` with methods: analytical_shrinkage, mp, mp2,
  linear_shrinkage, average_oracle, oracle_mv, poc.

"""

import numpy as np
import scipy.optimize

class cleaning_methods:

    def __init__(self, X):
        self.X = X
        self.n, self.p = X.shape

    def analytical_shrinkage(self):
        """Shrinkage analytique de Ledoit & Wolf.

        Returns:
            Sigma_tilde: matrice de covariance corrigée (p x p)
        """
        
        X, n, p = self.X, self.n, self.p

        # Sample covariance
        std = X.std(axis=0, ddof=1)
        D = np.diag(std)
        
        # extraction de la diagonale pour travailler en corrélation, reconstruction de la covariance en fin de fonction
        sample = np.corrcoef(X, rowvar = False)

        lambda_, u = np.linalg.eigh(sample)
        # tri des valeurs propres 
        idx = np.argsort(lambda_)
        lambda_ = lambda_[idx]
        u = u[:, idx]

        # nombre de valeurs propres proches de zero, au bruit près 
        eps = 1e-10
        mask = lambda_ > eps
        lambda_ = lambda_[mask]
        u = u[:, mask]
        m = len(lambda_) 

        L = np.tile(lambda_[:, None], (1, m))

        # Bandwidth
        h = n ** (-1 / 3)

        # Equation (4.9)
        H = h * L.T
        x = (L - L.T) / H

        ftilde = (3 / (4 * np.sqrt(5))) * np.mean(
            np.maximum(1 - x**2 / 5, 0) / H, axis=1
        )

        # Hilbert transform
        Hftemp = (
            (-3 / (10 * np.pi)) * x
            + (3 / (4 * np.sqrt(5) * np.pi))
            * (1 - x**2 / 5)
            * np.log(np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x)))
        )

        mask = np.abs(x) == np.sqrt(5)
        Hftemp[mask] = (-3 / (10 * np.pi)) * x[mask]

        Hftilde = np.mean(Hftemp / H, axis=1)

        # Shrunk eigenvalues
        if p <= n:
            dtilde = lambda_ / (
                (np.pi * (p / n) * lambda_ * ftilde) ** 2
                + (1 - (p / n) - np.pi * (p / n) * lambda_ * Hftilde) ** 2
            )
            dtilde = np.maximum(dtilde, 1e-8)
            R_tilde = u @ np.diag(dtilde)@ u.T
            Sigma_tilde = D @ R_tilde @ D       # reconstruction de la covariance
        else:
            # Equation (C.8)
            Hftilde0 = (
                (1 / np.pi)
                * (
                    3 / (10 * h**2)
                    + (3 / (4 * np.sqrt(5) * h))
                    * (1 - 1 / (5 * h**2))
                    * np.log((1 + np.sqrt(5) * h) / (1 - np.sqrt(5) * h))
                )
                * np.mean(1 / lambda_)
            )

            # Equation (C.5)
            dtilde0 = 1 / (np.pi * (p - m) / n * Hftilde0)

            dtilde1 = lambda_ / (
                np.pi**2 * lambda_**2 * (ftilde**2 + Hftilde**2)
            )
            
            R_tilde = dtilde0 * np.eye(p) + u @ np.diag(dtilde1 - dtilde0) @ u.T 
            Sigma_tilde = D @ R_tilde @ D       # reconstruction de la covariance
        
        return Sigma_tilde
    
    def mp(self):
        """Clipping des valeurs propres par quantile empiriqe (99%).

        Returns:
            matrice_cov_nettoyee: covariance nettoyée (p x p)
        """
        
        X = self.X
        n,p = np.shape(X)
        #lambda_max = (1 + np.sqrt(q))**2

        ecarts_types = np.std(X, axis=0)                # extraction de la diagonale pour travailler en corrélation, reconstruction de la covariance en fin de fonction
        D = np.diag(ecarts_types)
        matrice_corr = np.corrcoef(X, rowvar = False) 

        valeurs_p, vecteurs_p = np.linalg.eigh(matrice_corr) 
        lambda_max = np.percentile(valeurs_p, 99)       # borne supérieure pour le clipping prise au quantile 99 des valeurs propres empiriques
        
        indices_bruit = valeurs_p <= lambda_max
        moyenne_bruit = valeurs_p[indices_bruit].mean()
        valeurs_nettoyees = valeurs_p.copy()
        valeurs_nettoyees[indices_bruit] = moyenne_bruit        # imputation à leur moyenne des valeurs propres issues du bruit de type Wishart. 
        corr_propre = vecteurs_p @ np.diag(valeurs_nettoyees) @ vecteurs_p.T
        
        matrice_cov_nettoyee = D @ corr_propre @ D      # reconstruction de la covariance
        
        return matrice_cov_nettoyee
    
    def mp2(self):
        """Clipping des valeurs propres suivant la borne Marchenko-Pastur.

        Returns:
            matrice_cov_nettoyee: covariance nettoyée (p x p)
        """

        X = self.X
        n,p = np.shape(X)
        lambda_max = (1 + np.sqrt(p/n))**2          # borne supérieure pour le clipping prise selon la formule classique de Marchenko Pastur

        ecarts_types = np.std(X, axis=0)            # extraction de la diagonale pour travailler en corrélation, reconstruction de la covariance en fin de fonction
        D = np.diag(ecarts_types)
        matrice_corr = np.corrcoef(X, rowvar = False)

        valeurs_p, vecteurs_p = np.linalg.eigh(matrice_corr)

        indices_bruit = valeurs_p <= lambda_max
        moyenne_bruit = valeurs_p[indices_bruit].mean()
        valeurs_nettoyees = valeurs_p.copy()
        valeurs_nettoyees[indices_bruit] = moyenne_bruit         # imputation à leur moyenne des valeurs propres issues du bruit de type Wishart. 
        corr_propre = vecteurs_p @ np.diag(valeurs_nettoyees) @ vecteurs_p.T
        matrice_cov_nettoyee = D @ corr_propre @ D
        
        return matrice_cov_nettoyee
    
    def linear_shrinkage(self):
        """Shrinkage linéaire de Ledoit & Wolf (estimation analytique du facteur).

        Returns:
            matrice_shrinkee: matrice de covariance shrinkée (p x p)
        """

        X, n, p = self.X, self.n, self.p
        S = X.T @ X / (n-1)

        mn = np.trace(S) / p
        dn2 = (np.linalg.norm(S - mn * np.eye(p), 'fro')/np.sqrt(p))**2

        if dn2 == 0:
            return S

        bn2_barre = 0.0
        for k in range(n):
            xk = X[k, :].reshape(p, 1)
            Sk = xk @ xk.T
            bn2_barre += (np.linalg.norm(Sk - S, 'fro')**2)/p

        bn2_barre /= n**2
        bn2 = min(bn2_barre, dn2)
        an2 = dn2 - bn2

        return (bn2/dn2) * np.diag(np.diag(S)) + (an2/dn2) * S
    
    def average_oracle(self):
        """Oracle moyen : moyenne d'oracles locaux (fenêtres d'entraînement/test).

        Returns:
            sigma: estimation de la covariance reconstruite (p x p)
        """

        delta_train = 252 if self.n >= 1000 else int(self.n / 4)    # Définition des fenêtres d'entraînement et de test. delta_train = 252 pour s'assurer une structure de covariance constante sur la fenêtre.
        delta_test = 63 if self.n >= 1000 else int(self.n / 16)
        B = 100                                                     # Nombre d'oracles sur lesquels on fait la moyenne

        lambda_ao = np.zeros(self.p)

        X_prev = self.X[:delta_train]
        X_prev_centered = X_prev
        X_prev_norm = X_prev_centered / np.std(X_prev_centered, axis=0)             # !!!!!!!!!!!! on travaille en corrélation et on peut plus reconstruire 
        S_prev = X_prev_norm.T @ X_prev_norm / (delta_train - 1)

        eigval_prev, V_prev = np.linalg.eigh(S_prev)
        idx_prev = np.argsort(eigval_prev)[::-1]
        V_prev = V_prev[:, idx_prev]

        # temps sur lesquels on fait la moyenne
        t_b_list = np.random.randint(delta_train, self.n - delta_test, B)

        for t_b in t_b_list:

            X_b_prev = self.X[t_b-delta_train:t_b]  # fenêtre d'entrainement
            X_b_next = self.X[t_b:t_b+delta_test]   # fenêtre de test

            # covariances empiriques d'entraînement et de test 
            S_b_prev = X_b_prev.T @ X_b_prev / delta_train  
            S_b_next = X_b_next.T @ X_b_next / delta_test

            # Décompositions spectrales pour S_prev_b et S_next_b
            eigval_prev_b, V_prev_b = np.linalg.eigh(S_b_prev)
            idx_prev_b = np.argsort(eigval_prev_b)[::-1]
            V_prev_b = V_prev_b[:, idx_prev_b]

            eigval_next_b, V_next_b = np.linalg.eigh(S_b_next)
            idx_next_b = np.argsort(eigval_next_b)[::-1]
            eigval_next_b = eigval_next_b[idx_next_b]
            V_next_b = V_next_b[:, idx_next_b]

            # Matrice d'overlap
            H = V_prev_b.T @ V_next_b

            # Valeurs propres de l'oracle sur la fenêtre 
            lambda_o_b = (H**2) @ eigval_next_b

            # Moyenne
            lambda_ao += lambda_o_b / B

        xi = V_prev @ np.diag(lambda_ao) @ V_prev.T
        D = np.diag(np.sqrt(np.diag(S_prev)))           # reconstruction de la matrice de covariance
        sigma = D @ xi @ D

        return sigma
        
    def oracle_mv(self, Sigma):
        """Estimateur oracle MV (non bona fide) donné une matrice cible Sigma.
        
        Args:
            Sigma: matrice de covariance cible (p x p)
        
        Returns:
            Sigma_star: matrice de covariance oracle reconstruite (p x p)
        """
        X = self.X
        n, p = X.shape
        
        # Extraction de la diagonale pour travailler en corrélation
        std = X.std(axis=0, ddof=1)
        D = np.diag(std)
        X_scaled = X / std
        
        # Corrélation empirique
        S = X_scaled.T @ X_scaled / (n-1)  # ✓ Correction 1: (n-1)
        
        # Décomposition spectrale de la corrélation empirique
        lambda_, U = np.linalg.eigh(S)
        idx = np.argsort(lambda_)[::-1]
        U = U[:, idx]
        
        # Conversion de Sigma en corrélation (même espace que U)
        std_true = np.sqrt(np.diag(Sigma))
        D_true = np.diag(std_true)
        R_true = np.linalg.inv(D_true) @ Sigma @ np.linalg.inv(D_true)  # ✓ Correction 2
        
        # Valeurs propres de l'oracle dans l'espace corrélation
        d_star = np.zeros(p)
        for i in range(p):
            ui = U[:, i]
            d_star[i] = ui.T @ R_true @ ui  # ✓ Correction 3: R_true au lieu de Sigma
        
        # Reconstruction
        R_star = U @ np.diag(d_star) @ U.T
        Sigma_star = D @ R_star @ D  # Retour en covariance avec variances empiriques
        
        return Sigma_star


    def estimate_nfactor_act(self, C=1):
        # X est ici (n_temps x p_actifs)
        X = self.X
        n, p = X.shape
        corr = np.corrcoef(X.T)
        evals = np.flip(np.linalg.eigvalsh(corr))
        evals_adj = np.zeros(p - 1)
        for i in range(p - 1):
            mi = (np.sum(1.0 / (evals[(i + 1) :] - evals[i])) + 4.0 / (evals[i + 1] - evals[i])) / (p - i)
            rho = (p - i) / (n - 1)
            evals_adj[i] = -1.0 / (-(1 - rho) / evals[i] + rho * mi)
        thres = 1.0 + np.sqrt(p / (n - 1)) * C
        return np.where(evals_adj > thres)[0][-1] + 1
 
    def POET(self,K=None,C=None):
        X = self.X
        Y = X.T
        p, n = Y.shape
        
        if K == None:
            K = self.estimate_nfactor_act(C=1) # On passe Y_input car estimate attend (n x p)
 
        if K > 0:
            Dd, V = np.linalg.eigh(Y.T @ Y)
            Dd = Dd[::-1]
            V = np.flip(V, axis=1)
            F = np.sqrt(n) * V[:, :K]
            LamPCA = Y @ F / n
            Lowrank = LamPCA @ LamPCA.T
            uhat = Y - LamPCA @ F.T
            rate = 1/np.sqrt(p) + np.sqrt(np.log(p)/n)
        else:
            uhat = Y
            Lowrank = np.zeros((p, p))
            rate = np.sqrt(np.log(p)/n)
 
        SuPCA = uhat @ uhat.T / n
        
        if C == None:
            C = self.POETCmin(K) + 0.1
 
        lambda_mat = np.zeros((p, p))
        for i in range(p):
            res_prod = uhat[i, :] * uhat[i:, :]
            std_prod = np.std(res_prod, axis=1, ddof=1)
            l_row = std_prod * rate * C
            lambda_mat[i, i:] = l_row
            lambda_mat[i:, i] = l_row
 
        Rthresh = np.sign(SuPCA) * np.maximum(np.abs(SuPCA) - lambda_mat, 0)
        np.fill_diagonal(Rthresh, np.diag(SuPCA))
        SigmaY = Rthresh + Lowrank
 
        return  SigmaY
 
    def POETCmin(self, K):
        X = self.X
        def f(c_test):
            # On appelle POET avec le K fixe et le C à tester
            SigmaY = self.POET(K=K, C=c_test)
            return np.min(np.linalg.eigvals(SigmaY))
 
        # Test rapide des bornes
        if f(50) * f(0) < 0:
            root = scipy.optimize.fsolve(f, 0.5)[0]
            return max(0, root)
        
        return 0