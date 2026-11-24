# How the NDI reprojection adjustment transforms were fitted

In the November 12 recording, the green dot is a corner detected in the camera image, the red dot is its original NDI projection, and the yellow dot is its corrected NDI projection. We used the green dots from early frames to estimate two fixed rigid transforms. A board-frame transform $B$ acts before the original transform chain, and a camera-frame transform $C$ acts after it. For each subsequent frame, the ECM pose determines how the board appears in the camera. The same fitted $B$ and $C$ predict all 35 yellow dots, including corners without a green detection.

## From a board corner to an image pixel

Write $T_{a\leftarrow b}$ for a homogeneous transform that maps coordinates from frame $b$ into frame $a$. Products act on column vectors from right to left. In the equations below, $n$ denotes the NDI tracker, $m$ the tracked marker, $r$ the robot base, $e$ the ECM, $c$ the camera and $b$ the board. Frame index $f$ selects the recorded ECM pose. The nominal board-to-camera transform is

$$
T_f =
\left(T_{n\leftarrow r}\,T_{r\leftarrow e}(f)\,T_{e\leftarrow c}\right)^{-1}
T_{n\leftarrow m}\,T_{m\leftarrow b}.
$$

The rightmost two factors locate the board in the tracker frame. The term inside parentheses locates the camera in the same frame. Its inverse expresses the board in camera coordinates. This recording used a stationary marker pose, a 16.8 mm square pitch and a configured marker origin of $(146,112,-13.93)$ mm in board coordinates.

Each corner ID $i$ supplies a known point $p_i=(x_i,y_i,0,1)^\top$ in metres. Let $\pi$ map a camera-space point to image pixels using the recorded intrinsics and distortion coefficients. For a point $(X,Y,Z)$ with positive depth, the camera model first forms $(X/Z,Y/Z)$, applies lens distortion, then scales by the focal lengths and adds the principal point. Keeping that camera model fixed gives the two predictions

$$
u^{\mathrm{red}}_{fi}=\pi(T_f p_i),
\qquad
u^{\mathrm{yellow}}_{fi}=\pi(C\,T_f\,B\,p_i).
$$

**Why can the pixel shift vary across the image?** Consider just a camera-axis depth adjustment $\Delta Z$. In the undistorted pinhole model, the horizontal shift is

$$
\Delta u=f_x X\left(\frac{1}{Z+\Delta Z}-\frac{1}{Z}\right).
$$

On the optical axis $X=0$, this adjustment produces zero horizontal shift. Away from that axis, the shift depends on both lateral position and depth. Thus one fixed 3D correction can generate a different pixel shift for every corner and viewpoint.

## The unknowns and the fitting objective

Each adjustment has six parameters. A rotation vector $\omega$ gives the rotation axis through its direction and the angle in radians through its length. OpenCV converts it to a rotation matrix using the Rodrigues formula. Translation $t$ is in metres. The solver estimates all 12 parameters together, in the following order:

$$
H(\omega,t)=
\begin{bmatrix}
\operatorname{Rodrigues}(\omega) & t\\
0_{1\times3} & 1
\end{bmatrix},
\quad B=H(\omega_B,t_B),\quad C=H(\omega_C,t_C),
\quad \theta=(\omega_B,t_B,\omega_C,t_C).
$$

For a training frame with a timestamp-matched ECM pose, a visible corner contributes one correspondence. Its green pixel $g_{fi}$ and its model point $p_i$ are matched by ChArUco ID. With $\hat u_{fi}(\theta)=\pi(C T_f B p_i)$, the pixel residual is

$$
r_{fi}(\theta)=g_{fi}-\hat u_{fi}(\theta),
\qquad d_{fi}(\theta)=\lVert r_{fi}(\theta)\rVert_2.
$$

We seek a common parameter vector that minimizes the sum of Huber losses over the training correspondences $\mathcal S$:

$$
\hat\theta=\arg\min_{\theta}\sum_{(f,i)\in\mathcal S}\rho_\delta(d_{fi}(\theta)),
\qquad
\rho_\delta(d)=
\begin{cases}
\tfrac12 d^2 & d\leq\delta,\\
\delta(d-\tfrac12\delta) & d>\delta,
\end{cases}
\quad \delta=5\ \mathrm{px}.
$$

The threshold applies to the Euclidean length of each two-component residual. Beyond 5 pixels the loss grows linearly, so a large detection error has limited influence. Those corners still contribute to the fit.

## The numerical update

1. **Initialize.** Set every entry of $\theta$ to zero. Both transforms are then identity matrices and the first predictions coincide with the red dots.
2. **Estimate pixel sensitivity.** Stack the predicted pixel coordinates into a vector $\hat u$ and the residuals into $r$. Estimate column $j$ of the prediction Jacobian by a forward difference, $J_{:j}\approx[\hat u(\theta+h e_j)-\hat u(\theta)]/h$, with $h=10^{-6}$ in the units of parameter $j$.
3. **Weight each corner.** Set $w_{fi}=\min(1,\delta/\max(d_{fi},10^{-8}))$. The diagonal weight matrix $W$ contains that weight twice, once per pixel axis. The code multiplies the corresponding rows of $J$ and $r$ by $\sqrt{w_{fi}}$.
4. **Compute a damped Gauss–Newton step.** A small change $\Delta\theta$ gives $r(\theta+\Delta\theta)\approx r-J\Delta\theta$. Minimizing the weighted squared length of this linearized residual with a damping term $100\lVert\Delta\theta\rVert_2^2$ yields

$$
\left(J^\top WJ+100I_{12}\right)\Delta\theta=J^\top Wr.
$$

5. **Accept a step and repeat.** Try $\theta+\alpha\Delta\theta$ with $\alpha$ equal to $1,1/2,1/4,1/8,1/16$, in order. Accept the first step that reduces the weighted sum of squared pixel residuals using the current weights. Recompute the weights at the next iteration. Stop when the accepted step norm is below $10^{-7}$, no trial improves the fit, or 35 iterations have run.

## Fitted transforms and a worked example

The resulting matrices are rounded here to six decimal places. Their last columns contain translations in metres. The full-precision vectors are stored in the joint model parameters in the locally generated `correction_study/study.json`.

$$
B\approx
\begin{bmatrix}
 0.998001 & 0.036676 & 0.051458 & -0.003512\\
-0.038337 & 0.998763 & 0.031672 &  0.004837\\
-0.050233 &-0.033582 & 0.998173 & -0.017079\\
0&0&0&1
\end{bmatrix},
\qquad
C\approx
\begin{bmatrix}
 0.993640 & 0.109530 & 0.026128 & -0.009075\\
-0.108160 & 0.992921 &-0.049092 &  0.003536\\
-0.031320 & 0.045954 & 0.998452 & -0.032742\\
0&0&0&1
\end{bmatrix}.
$$

In held-out frame 3292, corner ID 0 has board coordinates $(0.0168,0.0840,0)$ m. Applying the two projection equations to this corner gives the following comparison with its observed pixel.

| Pixel position | Horizontal (px) | Vertical (px) | Distance to green (px) |
|---|---:|---:|---:|
| Green observation | 1188.46 | 391.44 | 0 |
| Red prediction | 1244.06 | 574.90 | 191.70 |
| Yellow prediction | 1198.45 | 387.48 | 10.75 |

For this corner the correction moves the prediction about 45.61 px left and 187.42 px up. The solver learned the transforms from earlier frames. The green pixel in frame 3292 measures their prediction error. The rendering script saves this example as `correction_study/held_out_frame_3292.jpg`.

## Model selection and interpretation

Training used every fifth frame from frames 422–2503, yielding 10,387 observed corner correspondences. Validation used frames 2534–3261 to compare a constant pixel shift, an affine image correction, camera-only and board-only rigid adjustments, and the joint fit. The joint fit had the lowest validation pixel RMSE at 14.35 px. The final test used frames 3292–4208. A 30-frame gap preceded each evaluation interval. Frames using a held ECM pose were excluded from fitting and scoring. [Study implementation](scripts/study_correction.py)

Reported pixel RMSE uses the ordinary Euclidean pixel distances, including large residuals. For $N$ evaluated correspondences,

$$
\mathrm{RMSE}_{\mathrm{px}}=
\sqrt{\frac{1}{N}\sum_{(f,i)}\lVert g_{fi}-\hat u_{fi}\rVert_2^2}.
$$

Across 24,380 held-out corner observations, RMSE fell from **146.58 px to 13.45 px**. A separate 3D comparison against the vision-estimated board pose gave **22.61 mm to 1.95 mm RMSE** on corners with a valid PnP estimate. The 3D comparison measures agreement with a board pose inferred from image detections and the assumed board dimensions.

The experiment identifies a correction to the combined transform chain. Because the fitting criterion sees only the final pixels, offsets in different parts of the chain can compensate for one another. For example, the fitted translation in $B$ alone does not establish a physical post height. The held-out result measures performance on later views of this same mounted board and camera setup.
