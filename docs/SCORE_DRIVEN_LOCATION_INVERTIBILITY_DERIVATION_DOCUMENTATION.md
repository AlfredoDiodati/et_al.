# sd/score_driven_location.h - the invertibility derivation

## Overview

The proof behind the invertibility condition `sd/score_driven_location.h` imposes, as it was written for this project, reproduced in full below. It is stated in its own notation: `x_t` for the observations, `c` for the mean, `mu_t` for the location in deviation from it, scalar autoregressive coefficients `phi_i` with `p` lags, `q` score lags with loadings `Psi_i`, and `u(v)` for the score rescaled by `nu/(nu+K) Sigma`. `docs/SCORE_DRIVEN_LOCATION_INVERTIBILITY_DOCUMENTATION.md` maps that notation onto the header's (its `q` is the header's `p`, its `Psi_j` is `sqrt((nu+K)(nu+2))/nu A_j L^-T`, and the header's diagonal `b` is Remark 2 with `Phi_1 = diag(b)`), states which parts the header imposes and which it does not, and records what was measured.

One part of Section 5 is not what the header implements: the header applies the radial map (49)-(52) to `(b, A)` rather than to `(phi, Psi~)`, so that a diagonal `B` and diagonal `A_j` stay diagonal, and it does not impose the bounds (46)-(48) on `L` and `nu`. The reasons are in the invertibility file.

## Invertibility of the filter of the multivariate Student-t score-driven location model

Reference. [B] F. Blasques (2019), *Advanced Econometric Methods: A Guide to Estimation and Inference for Complex Dynamic Models*.

Results of [B] used: Definition 5.1 (filter invertibility); Definition 5.2 (uniformly invertible filter); Theorem 4.23 (uniform contraction theorem, §4.3); Remark 5.1 and §5.3 (application of contraction theorems to the filtering equation, with the data as exogenous innovations); §4.6 (applicability of the Chapter 4 theorems to multivariate recursions). Section 5.3 of [B] refers to the uniform contraction theorem as Theorem 4.24; here Theorem 4.23 denotes the statement so labelled in §4.3. For a state in $\mathbb{R}^d$ equipped with a norm $\|\cdot\|_\star$, condition A3 of Theorem 4.23 is read as $\sup_{(x,\varepsilon)}\|\partial\phi(x,\varepsilon)/\partial x'\|_\star<1$, with $\|\cdot\|_\star$ the induced operator norm.

---

### 1. Model, filter and notation

Let $K,p,q\in\mathbb{N}$, $p,q\ge 1$, and $m=\max(p,q)$. The parameter is $\theta=(c,\phi_1,\dots,\phi_p,\Psi_1,\dots,\Psi_q,L,\nu)$ with $c\in\mathbb{R}^K$, $\phi_i\in\mathbb{R}$, $\Psi_i\in\mathbb{R}^{K\times K}$, $L\in\mathbb{R}^{K\times K}$ lower triangular with strictly positive diagonal, and $\nu>0$. Set $\phi_i=0$ for $p<i\le m$ and $\Psi_i=0$ for $q<i\le m$. The scale matrix is

$$\Sigma = LL' \tag{1}$$

For $v\in\mathbb{R}^K$ define

$$w(v) = \frac{\nu}{\nu+v'\Sigma^{-1}v} \tag{2}$$

$$u(v) = w(v)\,v \tag{3}$$

Model. The observations $x_t\in\mathbb{R}^K$ satisfy

$$x_t = c+\mu_t+v_t \tag{4}$$

$$v_t\sim t_K(0,\Sigma,\nu)\ \text{i.i.d.} \tag{5}$$

$$\mu_{t+1} = \sum_{i=1}^{m}\left[\phi_i\,\mu_{t+1-i}+\Psi_i\,u(v_{t+1-i})\right] \tag{6}$$

where $t_K(0,\Sigma,\nu)$ is the $K$-variate Student-t distribution with location $0$, scale matrix $\Sigma$ and $\nu$ degrees of freedom. With $f$ the conditional density of $x_t$ given $\mu_t$,

$$\ln f(x_t\mid\mu_t) = \kappa_0 - \tfrac{1}{2}\ln|\Sigma| - \tfrac{\nu+K}{2}\ln\!\left(1+\nu^{-1}v_t'\Sigma^{-1}v_t\right) \tag{7}$$

where $\kappa_0$ does not depend on $\mu_t$. Differentiating (7),

$$\frac{\partial\ln f(x_t\mid\mu_t)}{\partial\mu_t} = \frac{\nu+K}{\nu}\,\Sigma^{-1}u(v_t) \tag{8}$$

so $u(v_t)$ is the score of $\mu_t$ premultiplied by $\frac{\nu}{\nu+K}\Sigma$.

Filter. For given data $\{x_t\}$ and initial values $\hat\mu_1,\hat\mu_0,\dots,\hat\mu_{2-m}\in\mathbb{R}^K$, the filter is defined for $t\ge 1$ by

$$\hat\mu_{t+1} = \sum_{i=1}^{m}\left[\phi_i\,\hat\mu_{t+1-i}+\Psi_i\,u\!\left(x_{t+1-i}-c-\hat\mu_{t+1-i}\right)\right] \tag{9}$$

Let $\hat X_t=(\hat\mu_t',\dots,\hat\mu_{t-m+1}')'\in\mathbb{R}^{mK}$, with $k$-th block $\hat X_{t,k}=\hat\mu_{t-k+1}$, and $\xi_t=(x_t',\dots,x_{t-m+1}')'$. Then (9) is

$$\hat X_{t+1} = g(\hat X_t,\xi_t;\theta) \tag{10}$$

where the first block of $g(X,\xi;\theta)$ is $\sum_{i=1}^m[\phi_iX_i+\Psi_iu(\xi_i-c-X_i)]$ and its $k$-th block is $X_{k-1}$ for $2\le k\le m$. Write $\hat X_t(\theta,\hat X_1)$ for the solution of (10) started at $\hat X_1$.

Notation. $\|\cdot\|$ is the Euclidean norm on $\mathbb{R}^K$ and the spectral norm on $\mathbb{R}^{K\times K}$. Define

$$\tilde\Psi_i = L^{-1}\Psi_iL \tag{11}$$

$$\kappa_i(\theta) = \left\|\phi_iI_K-\tfrac{7}{16}\tilde\Psi_i\right\|+\tfrac{9}{16}\left\|\tilde\Psi_i\right\| \tag{12}$$

$$C(\theta) = \sum_{i=1}^{m}\kappa_i(\theta) \tag{13}$$

$$\|a\|_L = \|L^{-1}a\|,\quad a\in\mathbb{R}^K \tag{14}$$

$$\|X\|_\rho = \max_{1\le k\le m}\rho^{k-1}\|X_k\|_L,\quad X\in\mathbb{R}^{mK},\ \rho\in(0,1] \tag{15}$$

$$\|X\|_\infty = \max_{1\le k\le m}\|X_k\|,\quad X\in\mathbb{R}^{mK} \tag{16}$$

Assumptions.

(S) $\{x_t\}_{t\in\mathbb{Z}}$ is strictly stationary and ergodic.

(P) $\Theta$ is a set of parameters as above for which $\bar\ell=\sup_\Theta\|L\|$, $\underline\ell=\sup_\Theta\|L^{-1}\|$, $\bar\nu=\sup_\Theta\nu$ and $\bar\psi=\sup_\Theta\max_i\|\tilde\Psi_i\|$ are finite, and for some $\delta\in(0,1)$

$$\sup_{\theta\in\Theta}C(\theta)\le 1-\delta \tag{17}$$

Constants. For $m\ge 2$ let

$$\rho = (1-\delta/2)^{1/(m-1)} \tag{18}$$

$$\gamma = \max\left(\frac{1-\delta}{1-\delta/2},\,\rho\right) \tag{19}$$

and for $m=1$ let $\rho=1$ and $\gamma=1-\delta$. In both cases $\gamma\in(0,1)$, and $\rho,\gamma$ depend only on $(m,\delta)$.

---

### 2. Lemmas

Lemma 1. Let $v\in\mathbb{R}^K$, $z=L^{-1}v$ and $D(v)=\partial u(v)/\partial v'$. Then

$$N(z) := L^{-1}D(v)L = w\left(I_K-\frac{2w}{\nu}zz'\right),\quad w=\frac{\nu}{\nu+z'z} \tag{20}$$

$N(z)$ is symmetric with spectrum contained in $[-1/8,1]$, and

$$\left\|N(z)-\tfrac{7}{16}I_K\right\|\le\tfrac{9}{16} \tag{21}$$

Proof. By (1), $v'\Sigma^{-1}v=z'z$. From (2),

$$\frac{\partial w(v)}{\partial v'} = -\frac{2w(v)^2}{\nu}\,v'\Sigma^{-1} \tag{22}$$

and from (3) and (22),

$$D(v) = w(v)I_K-\frac{2w(v)^2}{\nu}\,vv'\Sigma^{-1} \tag{23}$$

Since $\Sigma^{-1}L=L^{-T}$,

$$L^{-1}vv'\Sigma^{-1}L = zz' \tag{24}$$

and (20) follows from (23) and (24). Let $s=z'z$. On $\mathrm{span}(z)^\perp$, $N(z)$ acts as $w\in(0,1]$. On $\mathrm{span}(z)$ it acts as

$$h(s) = \frac{\nu(\nu-s)}{(\nu+s)^2} \tag{25}$$

with

$$h'(s) = \frac{\nu(s-3\nu)}{(\nu+s)^3} \tag{26}$$

Hence $h$ is decreasing on $[0,3\nu]$ and increasing on $[3\nu,\infty)$, with $h(0)=1$, $h(3\nu)=-1/8$ and $h(s)\to 0$ as $s\to\infty$, so $h([0,\infty))\subset[-1/8,1]$. The spectrum of the symmetric matrix $N(z)-\frac{7}{16}I_K$ is therefore contained in $[-9/16,9/16]$, which gives (21). ∎

Lemma 2. For every $v\in\mathbb{R}^K$,

$$\|u(v)\|_L\le\tfrac{\sqrt\nu}{2} \tag{27}$$

Proof. With $z=L^{-1}v$, $\|u(v)\|_L=\nu\|z\|/(\nu+\|z\|^2)$, and $\nu+\|z\|^2\ge 2\sqrt\nu\,\|z\|$. ∎

Lemma 3. For every $\theta$ with $C(\theta)\le 1-\delta$, every $\xi\in\mathbb{R}^{mK}$ and every $X\in\mathbb{R}^{mK}$,

$$\left\|\frac{\partial g(X,\xi;\theta)}{\partial X'}\right\|_\rho\le\gamma \tag{28}$$

and, for all $X,X'\in\mathbb{R}^{mK}$,

$$\|g(X,\xi;\theta)-g(X',\xi;\theta)\|_\rho\le\gamma\,\|X-X'\|_\rho \tag{29}$$

Proof. Let $v_i=\xi_i-c-X_i$ and $z_i=L^{-1}v_i$. The Jacobian $J=\partial g/\partial X'$ has first block row $(A_1,\dots,A_m)$ with

$$A_i = \phi_iI_K-\Psi_iD(v_i) \tag{30}$$

and $k$-th block row, $2\le k\le m$, equal to $I_K$ in block column $k-1$ and zero elsewhere. By (11) and (20),

$$L^{-1}A_iL = \left(\phi_iI_K-\tfrac{7}{16}\tilde\Psi_i\right)-\tilde\Psi_i\left(N(z_i)-\tfrac{7}{16}I_K\right) \tag{31}$$

so, by (21) and (12),

$$\|L^{-1}A_iL\|\le\kappa_i(\theta) \tag{32}$$

Let $Y\in\mathbb{R}^{mK}$. Since $\|Y_i\|_L\le\rho^{-(i-1)}\|Y\|_\rho$ and $\rho\le 1$, (32) gives

$$\|(JY)_1\|_L\le\rho^{-(m-1)}C(\theta)\,\|Y\|_\rho \tag{33}$$

By (18), $\rho^{-(m-1)}C(\theta)\le(1-\delta)/(1-\delta/2)$ for $m\ge 2$, and $\rho^{-(m-1)}C(\theta)\le 1-\delta$ for $m=1$. For $2\le k\le m$,

$$\rho^{k-1}\|(JY)_k\|_L = \rho\cdot\rho^{k-2}\|Y_{k-1}\|_L\le\rho\,\|Y\|_\rho \tag{34}$$

By (15) and (19), $\|JY\|_\rho\le\gamma\|Y\|_\rho$, which is (28). As $g(\cdot,\xi;\theta)$ is continuously differentiable on the convex set $\mathbb{R}^{mK}$, (29) follows from (28) by the mean value inequality. ∎

Lemma 4. For every $\theta$ and every $i$,

$$\kappa_i(\theta)\ge|\phi_i| \tag{35}$$

Consequently, $C(\theta)<1$ implies $\sum_{i=1}^p|\phi_i|<1$, and then all roots of $1-\sum_{i=1}^p\phi_iz^i$ lie outside the unit circle.

Proof. Let $e\in\mathbb{R}^K$ with $\|e\|=1$ and $a=e'\tilde\Psi_ie$. Since $\|A\|\ge|e'Ae|$ for $A\in\mathbb{R}^{K\times K}$,

$$\kappa_i(\theta)\ge\left|\phi_i-\tfrac{7}{16}a\right|+\tfrac{9}{16}|a| = \max\left(|\phi_i-a|,\,|\phi_i+a/8|\right) \tag{36}$$

If $\phi_ia\ge 0$, then $|\phi_i+a/8|\ge|\phi_i|$; if $\phi_ia<0$, then $|\phi_i-a|\ge|\phi_i|$. This proves (35). The remaining statements follow from (13), (35) and $|\sum_{i=1}^p\phi_iz^i|\le\sum_{i=1}^p|\phi_i|<1$ for $|z|\le 1$. ∎

---

### 3. Invertibility

Theorem. Assume (S) and (P).

(a) For every $\theta\in\Theta$ there exists a unique strictly stationary and ergodic sequence $\{X_t(\theta)\}_{t\in\mathbb{Z}}$ such that, for every $\hat X_1\in\mathbb{R}^{mK}$, $\hat X_t(\theta,\hat X_1)\to X_t(\theta)$ exponentially fast almost surely; moreover $E\|X_t(\theta)\|^n<\infty$ for every $n>0$. The filter (9) is invertible at every $\theta\in\Theta$ in the sense of Definition 5.1 of [B].

(b) There exists a version $\{\tilde X_t(\theta)\}$ of the limit in (a) such that, for every $\hat X_1\in\mathbb{R}^{mK}$ and $t\ge 1$,

$$\sup_{\theta\in\Theta}\left\|\hat X_t(\theta,\hat X_1)-\tilde X_t(\theta)\right\|_\infty\le\bar\ell\,\rho^{-(m-1)}\gamma^{t-1}\left(\underline\ell\,\|\hat X_1\|_\infty+\frac{\bar B}{1-\gamma}\right) \tag{37}$$

where

$$\bar B = \tfrac{1}{2}\sqrt{\bar\nu}\,m\,\bar\psi \tag{38}$$

The filter (9) is exponentially fast almost surely uniformly invertible on $\Theta$ in the sense of Definition 5.2 of [B].

Proof of (a). Fix $\theta\in\Theta$. By Remark 5.1 and §5.3 of [B], it suffices to verify the conditions of Theorem 4.23 of [B] for the recursion (10), with state $X\in\mathbb{R}^{mK}$ normed by (15) (§4.6 of [B]) and innovations $\xi_t$.

A1. $\xi_t$ is a measurable function of finitely many lags of $x_t$; by (S), $\{\xi_t\}_{t\in\mathbb{Z}}$ is an exogenous strictly stationary and ergodic $mK$-variate sequence.

A2. The first block of $g(0,\xi_t;\theta)$ is $\sum_i\Psi_iu(x_{t+1-i}-c)$ and the other blocks are zero. Since $\|\Psi_ia\|_L\le\|\tilde\Psi_i\|\,\|a\|_L$, Lemma 2 gives, for every realization,

$$\|g(0,\xi_t;\theta)\|_\rho\le B(\theta) := \tfrac{1}{2}\sqrt\nu\sum_{i=1}^m\|\tilde\Psi_i\| \tag{39}$$

Hence $E\|g(0,\xi_1;\theta)\|_\rho^n<\infty$ for every $n>0$.

A3. By (17) and Lemma 3, $\sup_{(X,\xi)}\|\partial g(X,\xi;\theta)/\partial X'\|_\rho\le\gamma<1$.

Theorem 4.23 of [B] yields the claims of (a) in the norm $\|\cdot\|_\rho$, and they transfer to any norm on $\mathbb{R}^{mK}$ by equivalence of norms. Convergence for every $\hat X_1$ is the defining property in Definition 5.1 of [B]. ∎

Proof of (b). Fix $\theta\in\Theta$, write $g_s=g(\cdot,\xi_s;\theta)$, and define for $k\ge 1$

$$X_t^{(k)}(\theta) = g_{t-1}\circ g_{t-2}\circ\cdots\circ g_{t-k}(0) \tag{40}$$

By (29) and (39), for every realization,

$$\left\|X_t^{(k+1)}(\theta)-X_t^{(k)}(\theta)\right\|_\rho\le\gamma^kB(\theta) \tag{41}$$

Hence $\tilde X_t(\theta)=\lim_{k\to\infty}X_t^{(k)}(\theta)$ exists for every realization, and, since $X_t^{(1)}(\theta)=g_{t-1}(0)$,

$$\left\|\tilde X_t(\theta)\right\|_\rho\le\frac{B(\theta)}{1-\gamma} \tag{42}$$

Letting $k\to\infty$ in $X_{t+1}^{(k+1)}(\theta)=g_t(X_t^{(k)}(\theta))$ and using continuity of $g_t$ gives $\tilde X_{t+1}(\theta)=g_t(\tilde X_t(\theta))$. Applying (29) $t-1$ times,

$$\left\|\hat X_t(\theta,\hat X_1)-\tilde X_t(\theta)\right\|_\rho\le\gamma^{t-1}\left\|\hat X_1-\tilde X_1(\theta)\right\|_\rho \tag{43}$$

By (43) and (a), $\tilde X_t(\theta)=X_t(\theta)$ almost surely for every $t$, so $\{\tilde X_t(\theta)\}$ is a version of the limit in (a). By (14)–(16),

$$\|X\|_\rho\le\|L^{-1}\|\,\|X\|_\infty \tag{44}$$

$$\|X\|_\infty\le\|L\|\,\rho^{-(m-1)}\|X\|_\rho \tag{45}$$

By (P), $\|L\|\le\bar\ell$, $\|L^{-1}\|\le\underline\ell$ and $B(\theta)\le\bar B$ on $\Theta$. Combining (42)–(45) and taking the supremum over $\Theta$ gives (37). Its right-hand side does not depend on $\theta$ or on the realization and is $O(\gamma^t)$, which is exponentially fast almost sure uniform invertibility in the sense of Definition 5.2 of [B]. ∎

---

### 4. Remarks

1. For $K=1$ and $m=1$, the identity $|a|+|b|=\max(|a+b|,|a-b|)$ gives $C(\theta)=\max(|\phi_1-\Psi_1|,\,|\phi_1+\Psi_1/8|)$.

2. If $\phi_iI_K$ is replaced by a matrix $\Phi_i\in\mathbb{R}^{K\times K}$ in (6) and (9), the results hold with $\phi_iI_K$ replaced by $L^{-1}\Phi_iL$ in (12) and (31). Lemma 4 is not used in Section 3.

3. Condition (17) is sufficient and not necessary for invertibility.

4. By Lemma 4, (17) implies $\sum_{i=1}^p|\phi_i|<1$. Under (5), $\{u(v_t)\}$ is i.i.d. and bounded by Lemma 2, so the location $\mu_t$ generated by (6) is covariance stationary under (17).

---

### 5. A parametrization satisfying (P)

Fix $\delta\in(0,1)$, $0<\varepsilon_L<L_{\max}$, $b_L>0$ and $0<\nu_{\min}<\nu_{\max}$. Let $\sigma(a)=(1+e^{-a})^{-1}$. The unconstrained parameter is $\vartheta=(c,\phi^\circ_1,\dots,\phi^\circ_p,\tilde\Psi^\circ_1,\dots,\tilde\Psi^\circ_q,\ell^\circ,\nu^\circ)$ with $c\in\mathbb{R}^K$, $\phi^\circ_i\in\mathbb{R}$, $\tilde\Psi^\circ_i\in\mathbb{R}^{K\times K}$, $\ell^\circ=(\ell^\circ_{kl})_{k\ge l}\in\mathbb{R}^{K(K+1)/2}$ and $\nu^\circ\in\mathbb{R}$. Set $\phi^\circ_i=0$ for $p<i\le m$ and $\tilde\Psi^\circ_i=0$ for $q<i\le m$. The map $\vartheta\mapsto\theta$ is defined as follows.

For $1\le k\le K$,

$$L_{kk} = \varepsilon_L+(L_{\max}-\varepsilon_L)\,\sigma(\ell^\circ_{kk}) \tag{46}$$

for $k>l$,

$$L_{kl} = b_L\tanh(\ell^\circ_{kl}) \tag{47}$$

and

$$\nu = \nu_{\min}+(\nu_{\max}-\nu_{\min})\,\sigma(\nu^\circ) \tag{48}$$

Let

$$C^\circ = \sum_{i=1}^m\left(\left\|\phi^\circ_iI_K-\tfrac{7}{16}\tilde\Psi^\circ_i\right\|+\tfrac{9}{16}\left\|\tilde\Psi^\circ_i\right\|\right) \tag{49}$$

$$\lambda = (1-\delta)\,\frac{\tanh(C^\circ)}{C^\circ} \tag{50}$$

with $\lambda=1-\delta$ if $C^\circ=0$. Then

$$\phi_i = \lambda\,\phi^\circ_i \tag{51}$$

$$\tilde\Psi_i = \lambda\,\tilde\Psi^\circ_i \tag{52}$$

$$\Psi_i = L\tilde\Psi_iL^{-1} \tag{53}$$

and $c$ is unrestricted.

Proposition. Let $\Theta$ be the image of the map (46)–(53).

(i) For every $\vartheta$,

$$C(\theta) = (1-\delta)\tanh(C^\circ) \tag{54}$$

(ii) $\Theta$ satisfies (P) with constant $\delta$.

(iii) The map $(\phi^\circ,\tilde\Psi^\circ)\mapsto(\phi,\tilde\Psi)$ defined by (49)–(52) is a bijection from $\mathbb{R}^p\times(\mathbb{R}^{K\times K})^q$ onto $\{(\phi,\tilde\Psi):C<1-\delta\}$, with inverse

$$(\phi^\circ,\tilde\Psi^\circ) = \frac{\operatorname{artanh}\!\left(C/(1-\delta)\right)}{C}\,(\phi,\tilde\Psi) \tag{55}$$

for $C>0$, and $(\phi^\circ,\tilde\Psi^\circ)=0$ for $C=0$.

Proof. By (11) and (53), $C(\theta)$ is (13) evaluated at $(\phi,\tilde\Psi)$. Each $\kappa_i$ is positively homogeneous of degree one in $(\phi_i,\tilde\Psi_i)$, so (50)–(52) give $C(\theta)=\lambda C^\circ$, which is (54). For (ii): by (54), $C(\theta)<1-\delta$; $\|\tilde\Psi_i\|\le\frac{16}{9}C(\theta)<\frac{16}{9}$; $\nu\le\nu_{\max}$; $\|L\|$ is bounded because its entries are bounded by (46) and (47); and $\|L^{-1}\|$ is bounded because $L^{-1}$ is lower triangular, and by forward substitution its entries are polynomials in $(L_{kl})_{k>l}$ and $(L_{kk}^{-1})$, which are bounded by (46) and (47). For (iii): $C=0$ if and only if $(\phi,\tilde\Psi)=0$, since $C\ge\sum_i|\phi_i|$ by (35) and $C\ge\frac{9}{16}\sum_i\|\tilde\Psi_i\|$ by (12). The map preserves the direction of $(\phi^\circ,\tilde\Psi^\circ)$ because $\lambda>0$, and by (54) it maps $C^\circ\in[0,\infty)$ to $C\in[0,1-\delta)$ through the strictly increasing bijection $s\mapsto(1-\delta)\tanh(s)$. Solving (54) for $C^\circ$ and inverting (51)–(52) gives (55). ∎

Implementation. The likelihood at $\vartheta$ is evaluated by computing $\theta$ from (46)–(53) and running (9). Starting values follow from a candidate $\theta_0$ with $C(\theta_0)<1-\delta$ through (55), with $\tilde\Psi_{i,0}=L_0^{-1}\Psi_{i,0}L_0$, and through the inverses of (46)–(48). By Lemma 4, the parametrization requires no further restriction for stationarity of the autoregressive part. The map (49)–(52) is continuous, and it is differentiable except where the two largest singular values of one of the matrices in (49) coincide. Standard errors for $\theta$ follow from those of $\vartheta$ by the delta method applied to (46)–(53).
