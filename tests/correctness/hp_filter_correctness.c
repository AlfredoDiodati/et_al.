/*
Does filter/hp.h compute the Hodrick-Prescott trend and cycle.

Ported test (statsmodels 0.14.6, statsmodels/tsa/filters/tests/test_filters.py,
test_hpfilter): US real GDP, 1959Q1 to 2009Q3, 203 quarters from
statsmodels' macrodata dataset, lambda = 1600, against the cycle and trend
Stata computes, which statsmodels checks to 6 decimals. The data and the
Stata values below are copied from that dataset and that test.

Own checks:
- properties that hold in exact arithmetic for any lambda and any series:
  lambda = 0 gives the series back exactly; a straight line is its own
  trend; the cycle sums to zero and is orthogonal to the time index t,
  since 1 and t are in the null space of D; series of one or two
  observations are their own trend.
- the trend against a reference written here: the dense system
  (I + lambda D'D) tau = y built entry by entry and solved by Gaussian
  elimination in long double, for T of 3, 4, 10 and 200 and lambda of 0.5,
  1600 and 129600.
- linearity and constants (the trend of 2.5 a - 4 b + 7 is 2.5 trend(a)
  - 4 trend(b) + 7), time reversal (the matrix reads the same backwards),
  optimality (no perturbation of the trend lowers the objective, evaluated
  in long double), a larger lambda fitting worse and smoother, and the
  least squares line as lambda grows to 1e12 (float64);
- a matrix of series along each axis against the same series one at a time,
  a strided view, tensors on every axis and a permuted view, a NaN confined
  to its series, and a frame's structure.

The bound. The system is symmetric positive definite with condition number
at most 1 + 16 lambda, and banded LU is backward stable, so the trend is
within about (1 + 16 lambda) u times the size of the series, u the unit
roundoff of the build. The checks allow 64 (1 + 16 lambda) u max|y|. In
float32 at lambda = 1600 that is a few parts in a thousand, which is the
conditioning of the problem, not the method; the Stata comparison, whose
tolerance is statsmodels' 1.5e-6, runs in float64 only.
*/
#include "../../filter/hp.h"
#include "../check.h"
#include <stdio.h>

static const double unit_roundoff = sizeof(mreal) == sizeof(double) ? 1.1102230246251565e-16 : 5.9604644775390625e-08;

static double hp_bound(double lambda, const mreal *y, int n, ptrdiff_t stride) {
    double largest = 0;
    for (int t = 0; t < n; t++) if (fabs((double)y[t * stride]) > largest) largest = fabs((double)y[t * stride]);
    return 64 * (1 + 16 * lambda) * unit_roundoff * (largest > 1 ? largest : 1);
}

static const double realgdp[203] = {
    2710.349, 2778.801, 2775.488, 2785.204,
    2847.699, 2834.39, 2839.022, 2802.616,
    2819.264, 2872.005, 2918.419, 2977.83,
    3031.241, 3064.709, 3093.047, 3100.563,
    3141.087, 3180.447, 3240.332, 3264.967,
    3338.246, 3376.587, 3422.469, 3431.957,
    3516.251, 3563.96, 3636.285, 3724.014,
    3815.423, 3828.124, 3853.301, 3884.52,
    3918.74, 3919.556, 3950.826, 3980.97,
    4063.013, 4131.998, 4160.267, 4178.293,
    4244.1, 4256.46, 4283.378, 4263.261,
    4256.573, 4264.289, 4302.259, 4256.637,
    4374.016, 4398.829, 4433.943, 4446.264,
    4525.769, 4633.101, 4677.503, 4754.546,
    4876.166, 4932.571, 4906.252, 4953.05,
    4909.617, 4922.188, 4873.52, 4854.34,
    4795.295, 4831.942, 4913.328, 4977.511,
    5090.663, 5128.947, 5154.072, 5191.499,
    5251.762, 5356.131, 5451.921, 5450.793,
    5469.405, 5684.569, 5740.3, 5816.222,
    5825.949, 5831.418, 5873.335, 5889.495,
    5908.467, 5787.373, 5776.617, 5883.46,
    6005.717, 5957.795, 6030.184, 5955.062,
    5857.333, 5889.074, 5866.37, 5871.001,
    5944.02, 6077.619, 6197.468, 6325.574,
    6448.264, 6559.594, 6623.343, 6677.264,
    6740.275, 6797.344, 6903.523, 6955.918,
    7022.757, 7050.969, 7118.95, 7153.359,
    7193.019, 7269.51, 7332.558, 7458.022,
    7496.6, 7592.881, 7632.082, 7733.991,
    7806.603, 7865.016, 7927.393, 7944.697,
    8027.693, 8059.598, 8059.476, 7988.864,
    7950.164, 8003.822, 8037.538, 8069.046,
    8157.616, 8244.294, 8329.361, 8417.016,
    8432.485, 8486.435, 8531.108, 8643.769,
    8727.919, 8847.303, 8904.289, 9003.18,
    9025.267, 9044.668, 9120.684, 9184.275,
    9247.188, 9407.052, 9488.879, 9592.458,
    9666.235, 9809.551, 9932.672, 10008.874,
    10103.425, 10194.277, 10328.787, 10507.575,
    10601.179, 10684.049, 10819.914, 11014.254,
    11043.044, 11258.454, 11267.867, 11334.544,
    11297.171, 11371.251, 11340.075, 11380.128,
    11477.868, 11538.77, 11596.43, 11598.824,
    11645.819, 11738.706, 11935.461, 12042.817,
    12127.623, 12213.818, 12303.533, 12410.282,
    12534.113, 12587.535, 12683.153, 12748.699,
    12915.938, 12962.462, 12965.916, 13060.679,
    13099.901, 13203.977, 13321.109, 13391.249,
    13366.865, 13415.266, 13324.6, 13141.92,
    12925.41, 12901.504, 12990.341,
};
static const double stata_cycle[203] = {
    3.951191484487844718e01, 8.008853245681075350e01, 4.887545512195401898e01, 3.059193256079834100e01,
    6.488266733421960453e01, 2.304024204546703913e01, -1.355312369487364776e00, -6.746236512580753697e01,
    -8.136743836853429457e01, -6.016789026443257171e01, -4.636922433138215638e01, -2.069533915570400495e01,
    -2.162152558595607843e00, -4.718647774311648391e00, -1.355645669169007306e01, -4.436926204475639679e01,
    -4.332027378211660107e01, -4.454697106352068658e01, -2.629875787765286077e01, -4.426119635629265758e01,
    -1.443441190762496262e01, -2.026686669186437939e01, -1.913700136208899494e01, -5.482458977940950717e01,
    -1.596244517937793717e01, -1.374011542874541192e01, 1.325482813403914406e01, 5.603040174253828809e01,
    1.030743373627105939e02, 7.217534795943993231e01, 5.462972503693208637e01, 4.407065050666142270e01,
    3.749016270204992907e01, -1.511244199923112319e00, -9.093507374079763395e00, -1.685361946760258434e01,
    2.822211031434289907e01, 6.117590627896424849e01, 5.433135391434370831e01, 3.810480376716623141e01,
    7.042964928802848590e01, 4.996346842507591646e01, 4.455282059571254649e01, -7.584961950576143863e00,
    -4.620339247697120300e01, -7.054024364552969928e01, -6.492941099801464588e01, -1.433567024239555394e02,
    -5.932834493089012540e01, -6.842096758743628016e01, -6.774011924654860195e01, -9.030958565658056614e01,
    -4.603981499136807543e01, 2.588118806672991923e01, 3.489419371912299539e01, 7.675179642495095322e01,
    1.635497817724171910e02, 1.856079654765617306e02, 1.254269446392718237e02, 1.387413113837174024e02,
    6.201826599282230745e01, 4.122129542972197669e01, -4.120287475842360436e01, -9.486328233441963675e01,
    -1.894232132641573116e02, -1.895766639620087517e02, -1.464092413342650616e02, -1.218770668721217589e02,
    -4.973075629078175552e01, -5.365375213897277717e01, -7.175241524251214287e01, -7.834757283225462743e01,
    -6.264220687943907251e01, -3.054332122210325906e00, 4.808218808024685131e01, 2.781399326736391231e00,
    -2.197570415173231595e01, 1.509441335012807031e02, 1.658909029574851957e02, 2.027292548049981633e02,
    1.752101578176061594e02, 1.452808749847536092e02, 1.535481629475025329e02, 1.376169777998875361e02,
    1.257703080340770612e02, -2.524186846895645431e01, -6.546618027042404719e01, 1.192352023580315290e01,
    1.043482970188742911e02, 2.581376184768396342e01, 6.634330880534071184e01, -4.236780162594641297e01,
    -1.759397735321817891e02, -1.827933311233055065e02, -2.472312362505917918e02, -2.877470049336488955e02,
    -2.634066336693540507e02, -1.819572770763625158e02, -1.175034606274621183e02, -4.769898649718379602e01,
    1.419578280287896632e01, 6.267929662760798237e01, 6.196413196753746888e01, 5.019769125317907310e01,
    4.665364933213822951e01, 3.662430749527266016e01, 7.545680850246480986e01, 6.052940492147536133e01,
    6.029518881462354329e01, 2.187042136652689805e01, 2.380067926824722235e01, -7.119129802169481991e00,
    -3.194497359120850888e01, -1.897137038934124575e01, -1.832687287845146784e01, 4.600482336597542599e01,
    2.489047706403016491e01, 6.305909392127250612e01, 4.585212309498183458e01, 9.314260180878318351e01,
    1.129819097095369216e02, 1.204662123176703972e02, 1.336860614601246198e02, 1.034567175813735957e02,
    1.403118873372050075e02, 1.271726169351004501e02, 8.271925765282139764e01, -3.197432211752584408e01,
    -1.150209535194062482e02, -1.064694837456772802e02, -1.190428718925368230e02, -1.353635336292991269e02,
    -9.644348283027102298e01, -6.143413116116607853e01, -3.019161311097923317e01, 1.384333163552582846e00,
    -4.156016073666614830e01, -4.843882841860977351e01, -6.706442838867042155e01, -2.019644488579979225e01,
    -4.316446881084630149e00, 4.435061943264736328e01, 2.820550564155564643e01, 5.155624419490777655e01,
    -4.318760899315748247e00, -6.534632828542271454e01, -7.226757738268497633e01, -9.412378615444868046e01,
    -1.191240653288368776e02, -4.953669826751865912e01, -6.017251579067487910e01, -5.103438828313483100e01,
    -7.343057830678117170e01, -2.774245193054957781e01, -3.380481112519191811e00, -2.672779877794346248e01,
    -3.217342505148371856e01, -4.140567518359966925e01, -6.687756033938057953e00, 7.300600408459467872e01,
    6.862345670680042531e01, 5.497882461487461114e01, 9.612244093055960548e01, 1.978212770103891671e02,
    1.362772276848754700e02, 2.637635494867263333e02, 1.876813256815166824e02, 1.711447873158413131e02,
    5.257586460826678376e01, 4.710652228531762375e01, -6.237613484241046535e01, -9.982044354035315337e01,
    -7.916275548997509759e01, -9.526003459472303803e01, -1.147987680369169539e02, -1.900259054765901965e02,
    -2.212256473439556430e02, -2.071394278781845060e02, -8.968541528904825100e01, -6.189531564415665343e01,
    -5.662878162551714922e01, -4.961678134413705266e01, -3.836288992144181975e01, -8.956671991456460091e00,
    3.907028461866866564e01, 1.865299000184495526e01, 4.279803532226833340e01, 3.962735362631610769e01,
    1.412691291877854383e02, 1.256537791844366438e02, 7.067642758858892194e01, 1.108876647603192396e02,
    9.956490829291760747e01, 1.571612709880937473e02, 2.318746375812715996e02, 2.635546670125277160e02,
    2.044220965739259555e02, 2.213739418903714977e02, 1.020184547767112235e02, -1.072694716663390864e02,
    -3.490477058718843182e02, -3.975570728533530200e02, -3.331152428080622485e02,
};
static const double stata_trend[203] = {
    2.670837085155121713e03, 2.698712467543189177e03, 2.726612544878045810e03, 2.754612067439201837e03,
    2.782816332665780465e03, 2.811349757954532834e03, 2.840377312369487299e03, 2.870078365125807522e03,
    2.900631438368534418e03, 2.932172890264432681e03, 2.964788224331382025e03, 2.998525339155703932e03,
    3.033403152558595593e03, 3.069427647774311481e03, 3.106603456691690099e03, 3.144932262044756499e03,
    3.184407273782116590e03, 3.224993971063520803e03, 3.266630757877652741e03, 3.309228196356292756e03,
    3.352680411907625057e03, 3.396853866691864368e03, 3.441606001362089046e03, 3.486781589779409387e03,
    3.532213445179378141e03, 3.577700115428745448e03, 3.623030171865960710e03, 3.667983598257461836e03,
    3.712348662637289181e03, 3.755948652040559864e03, 3.798671274963067845e03, 3.840449349493338559e03,
    3.881249837297949853e03, 3.921067244199923152e03, 3.959919507374079785e03, 3.997823619467602384e03,
    4.034790889685657021e03, 4.070822093721035344e03, 4.105935646085656117e03, 4.140188196232833434e03,
    4.173670350711971878e03, 4.206496531574924120e03, 4.238825179404287155e03, 4.270845961950576566e03,
    4.302776392476971523e03, 4.334829243645529459e03, 4.367188410998014660e03, 4.399993702423955256e03,
    4.433344344930889747e03, 4.467249967587436004e03, 4.501683119246548813e03, 4.536573585656580690e03,
    4.571808814991368308e03, 4.607219811933269739e03, 4.642608806280876706e03, 4.677794203575049323e03,
    4.712616218227582976e03, 4.746963034523438182e03, 4.780825055360728584e03, 4.814308688616282780e03,
    4.847598734007177882e03, 4.880966704570278125e03, 4.914722874758424041e03, 4.949203282334419782e03,
    4.984718213264157384e03, 5.021518663962008759e03, 5.059737241334265491e03, 5.099388066872122181e03,
    5.140393756290781312e03, 5.182600752138972894e03, 5.225824415242512259e03, 5.269846572832254424e03,
    5.314404206879438789e03, 5.359185332122210639e03, 5.403838811919753425e03, 5.448011600673263274e03,
    5.491380704151732061e03, 5.533624866498719712e03, 5.574409097042514986e03, 5.613492745195001589e03,
    5.650738842182393455e03, 5.686137125015246056e03, 5.719786837052497503e03, 5.751878022200112355e03,
    5.782696691965922582e03, 5.812614868468956047e03, 5.842083180270424236e03, 5.871536479764196883e03,
    5.901368702981125352e03, 5.931981238152316109e03, 5.963840691194659485e03, 5.997429801625946311e03,
    6.033272773532181418e03, 6.071867331123305121e03, 6.113601236250591683e03, 6.158748004933649099e03,
    6.207426633669354487e03, 6.259576277076362203e03, 6.314971460627461965e03, 6.373272986497183410e03,
    6.434068217197121157e03, 6.496914703372392069e03, 6.561378868032462378e03, 6.627066308746821051e03,
    6.693621350667861407e03, 6.760719692504727391e03, 6.828066191497535328e03, 6.895388595078524304e03,
    6.962461811185376064e03, 7.029098578633473153e03, 7.095149320731752596e03, 7.160478129802169860e03,
    7.224963973591208742e03, 7.288481370389341464e03, 7.350884872878451461e03, 7.412017176634024509e03,
    7.471709522935970199e03, 7.529821906078727807e03, 7.586229876905018500e03, 7.640848398191216802e03,
    7.693621090290463144e03, 7.744549787682329224e03, 7.793706938539875409e03, 7.841240282418626521e03,
    7.887381112662795204e03, 7.932425383064899506e03, 7.976756742347178260e03, 8.020838322117525422e03,
    8.065184953519406008e03, 8.110291483745677397e03, 8.156580871892536379e03, 8.204409533629299403e03,
    8.254059482830271008e03, 8.305728131161165948e03, 8.359552613110980019e03, 8.415631666836447039e03,
    8.474045160736666730e03, 8.534873828418609264e03, 8.598172428388670596e03, 8.663965444885800025e03,
    8.732235446881084499e03, 8.802952380567352520e03, 8.876083494358445023e03, 8.951623755805092514e03,
    9.029585760899315574e03, 9.110014328285422380e03, 9.192951577382684263e03, 9.278398786154448317e03,
    9.366312065328836979e03, 9.456588698267518339e03, 9.549051515790675694e03, 9.643492388283135369e03,
    9.739665578306781754e03, 9.837293451930549054e03, 9.936052481112519672e03, 1.003560179877794326e04,
    1.013559842505148299e04, 1.023568267518359971e04, 1.033547475603393832e04, 1.043456899591540605e04,
    1.053255554329319966e04, 1.062907017538512628e04, 1.072379155906944106e04, 1.081643272298961165e04,
    1.090676677231512440e04, 1.099469045051327339e04, 1.108018567431848351e04, 1.116339921268415856e04,
    1.124459513539173349e04, 1.132414447771468258e04, 1.140245113484241119e04, 1.147994844354035376e04,
    1.155703075548997549e04, 1.163403003459472347e04, 1.171122876803691724e04, 1.178884990547659072e04,
    1.186704464734395515e04, 1.194584542787818464e04, 1.202514641528904758e04, 1.210471231564415575e04,
    1.218425178162551674e04, 1.226343478134413635e04, 1.234189588992144127e04, 1.241923867199145570e04,
    1.249504271538133071e04, 1.256888200999815490e04, 1.264035496467773191e04, 1.270907164637368442e04,
    1.277466887081221466e04, 1.283680822081556289e04, 1.289523957241141034e04, 1.294979133523968085e04,
    1.300033609170708223e04, 1.304681572901190702e04, 1.308923436241872878e04, 1.312769433298747208e04,
    1.316244290342607383e04, 1.319389205810962812e04, 1.322258154522328914e04, 1.324918947166633916e04,
    1.327445770587188417e04, 1.329906107285335383e04, 1.332345624280806260e04,
};

static void test_stata(void) {
    puts("statsmodels test_hpfilter: US real GDP, lambda 1600, against Stata");
    if (sizeof(mreal) != sizeof(double)) { puts("  (float64 only; skipped in this build)"); return; }
    Mat y = mat_new(203, 1);
    for (int t = 0; t < 203; t++) y.d[t] = (mreal)realgdp[t];
    Mat trend = mat_hp_trend(y, 1600, 0), cycle = mat_hp_cycle(y, 1600, 0);
    for (int t = 0; t < 203; t++) {
        CHECK_NEAR(trend.d[t], stata_trend[t], 1.5e-6, "trend against Stata");
        CHECK_NEAR(cycle.d[t], stata_cycle[t], 1.5e-6, "cycle against Stata");
    }
    mat_free(trend); mat_free(cycle); mat_free(y);
}

/* The dense system solved in long double. */
static void reference_trend(const mreal *y, int T, double lambda, long double *tau) {
    long double *a = calloc((size_t)T * T, sizeof(long double));
    for (int i = 0; i < T; i++) { a[(size_t)i * T + i] = 1; tau[i] = y[i]; }
    const int difference[3] = { 1, -2, 1 };
    for (int r = 0; r + 2 < T; r++)
        for (int p = 0; p < 3; p++)
            for (int q = 0; q < 3; q++) a[(size_t)(r + p) * T + r + q] += (long double)lambda * difference[p] * difference[q];
    for (int c = 0; c < T; c++) {
        int pivot = c;
        for (int r = c + 1; r < T; r++) if (fabsl(a[(size_t)r * T + c]) > fabsl(a[(size_t)pivot * T + c])) pivot = r;
        for (int j = 0; j < T; j++) { long double s = a[(size_t)c * T + j]; a[(size_t)c * T + j] = a[(size_t)pivot * T + j]; a[(size_t)pivot * T + j] = s; }
        long double s = tau[c]; tau[c] = tau[pivot]; tau[pivot] = s;
        for (int r = c + 1; r < T; r++) {
            long double f = a[(size_t)r * T + c] / a[(size_t)c * T + c];
            for (int j = c; j < T; j++) a[(size_t)r * T + j] -= f * a[(size_t)c * T + j];
            tau[r] -= f * tau[c];
        }
    }
    for (int c = T - 1; c >= 0; c--) {
        long double s = tau[c];
        for (int j = c + 1; j < T; j++) s -= a[(size_t)c * T + j] * tau[j];
        tau[c] = s / a[(size_t)c * T + c];
    }
    free(a);
}

static void test_properties_and_reference(void) {
    puts("lambda 0, straight lines, the cycle orthogonal to 1 and t, short series, and a long double reference");
    Rng rng = rng_new(1600, 0);
    int lengths[4] = { 3, 4, 10, 200 };
    double lambdas[3] = { 0.5, 1600, 129600 };
    for (int l = 0; l < 4; l++)
        for (int k = 0; k < 3; k++) {
            int T = lengths[l];
            double lambda = lambdas[k];
            Mat y = mat_new(T, 1);
            double level = 0;
            for (int t = 0; t < T; t++) { level += rng_normal(&rng); y.d[t] = (mreal)(100 + level); }
            double bound = hp_bound(lambda, y.d, T, 1);
            Mat trend = mat_hp_trend(y, lambda, 0), cycle = mat_hp_cycle(y, lambda, 0);
            long double *want = malloc((size_t)T * sizeof(long double));
            reference_trend(y.d, T, lambda, want);
            double sum = 0, weighted = 0;
            for (int t = 0; t < T; t++) {
                CHECK_NEAR(trend.d[t], want[t], bound, "trend against the long double reference");
                sum += (double)cycle.d[t];
                weighted += (double)cycle.d[t] * (t + 1);
            }
            CHECK_NEAR(sum, 0, T * bound, "cycle sums to zero");
            CHECK_NEAR(weighted, 0, T * T * bound, "cycle orthogonal to t");
            free(want); mat_free(trend); mat_free(cycle); mat_free(y);
        }

    Mat y = mat_new(50, 1);
    for (int t = 0; t < 50; t++) y.d[t] = (mreal)rng_normal(&rng);
    Mat same = mat_hp_trend(y, 0, 0);
    CHECK(memcmp(same.d, y.d, 50 * sizeof(mreal)) == 0, "lambda 0 returns the series");
    mat_free(same);
    for (int t = 0; t < 50; t++) y.d[t] = (mreal)(3 - 0.25 * t);
    Mat line = mat_hp_trend(y, 1600, 0);
    for (int t = 0; t < 50; t++) CHECK_NEAR(line.d[t], y.d[t], hp_bound(1600, y.d, 50, 1), "a straight line is its own trend");
    mat_free(line); mat_free(y);

    for (int T = 1; T <= 2; T++) {
        Mat s = mat_new(T, 1);
        for (int t = 0; t < T; t++) s.d[t] = (mreal)(7 + 3 * t);
        Mat trend = mat_hp_trend(s, 1600, 0);
        CHECK(memcmp(trend.d, s.d, (size_t)T * sizeof(mreal)) == 0, "a series of %d observations is its own trend", T);
        mat_free(trend); mat_free(s);
    }
}

/* The objective the trend minimises, in long double. */
static long double hp_objective(const mreal *y, const long double *tau, int T, double lambda) {
    long double fit = 0, rough = 0;
    for (int t = 0; t < T; t++) fit += ((long double)y[t] - tau[t]) * ((long double)y[t] - tau[t]);
    for (int t = 1; t + 1 < T; t++) {
        long double d = tau[t + 1] - 2 * tau[t] + tau[t - 1];
        rough += d * d;
    }
    return fit + (long double)lambda * rough;
}

static void test_structure(void) {
    puts("linearity, constants, time reversal, optimality, monotone in lambda, and the least squares line as lambda grows");
    Rng rng = rng_new(97, 0);
    int T = 150;
    Mat a = mat_new(T, 1), b = mat_new(T, 1), combined = mat_new(T, 1), reversed = mat_new(T, 1);
    for (int t = 0; t < T; t++) {
        a.d[t] = (mreal)(rng_normal(&rng) * 3 + 0.1 * t);
        b.d[t] = (mreal)rng_normal(&rng);
        combined.d[t] = (mreal)(2.5 * a.d[t] - 4 * b.d[t] + 7);
        reversed.d[T - 1 - t] = a.d[t];
    }
    double bound = hp_bound(1600, combined.d, T, 1);
    Mat ta = mat_hp_trend(a, 1600, 0), tb = mat_hp_trend(b, 1600, 0), tc = mat_hp_trend(combined, 1600, 0);
    Mat tr = mat_hp_trend(reversed, 1600, 0);
    for (int t = 0; t < T; t++) {
        CHECK_NEAR(tc.d[t], 2.5 * ta.d[t] - 4 * tb.d[t] + 7, bound, "linear, and a constant passes to the trend");
        CHECK_NEAR(tr.d[T - 1 - t], ta.d[t], hp_bound(1600, a.d, T, 1), "reversing the series reverses the trend");
    }

    /* no perturbation of the trend lowers the objective it minimises */
    long double *tau = malloc((size_t)T * sizeof(long double));
    for (int t = 0; t < T; t++) tau[t] = ta.d[t];
    long double best = hp_objective(a.d, tau, T, 1600);
    int lowered = 0;
    for (int trial = 0; trial < 200; trial++) {
        long double step = (long double)1e-3 * (trial % 2 ? 1 : -1);
        int k = rng_below(&rng, (uint64_t)T);
        tau[k] += step;
        if (hp_objective(a.d, tau, T, 1600) < best) lowered++;
        tau[k] -= step;
    }
    CHECK(lowered == 0, "%d of 200 perturbations lowered the objective", lowered);
    free(tau);

    /* a larger lambda fits worse and is smoother */
    double lambdas[4] = { 10, 100, 1600, 100000 }, last_fit = -1, last_rough = 1e300;
    for (int k = 0; k < 4; k++) {
        Mat trend = mat_hp_trend(a, lambdas[k], 0);
        double fit = 0, rough = 0;
        for (int t = 0; t < T; t++) fit += ((double)a.d[t] - trend.d[t]) * ((double)a.d[t] - trend.d[t]);
        for (int t = 1; t + 1 < T; t++) {
            double d = (double)trend.d[t + 1] - 2 * (double)trend.d[t] + (double)trend.d[t - 1];
            rough += d * d;
        }
        CHECK(fit > last_fit && rough < last_rough, "lambda %g: fit %g after %g, roughness %g after %g", lambdas[k], fit,
              last_fit, rough, last_rough);
        last_fit = fit;
        last_rough = rough;
        mat_free(trend);
    }

    /* as lambda grows the trend tends to the least squares line */
    if (sizeof(mreal) == sizeof(double)) {
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int t = 0; t < T; t++) { sx += t; sy += a.d[t]; sxx += (double)t * t; sxy += (double)t * a.d[t]; }
        double slope = (T * sxy - sx * sy) / (T * sxx - sx * sx), intercept = (sy - slope * sx) / T;
        Mat stiff = mat_hp_trend(a, 1e12, 0);
        double worst = 0;
        for (int t = 0; t < T; t++) {
            double gap = fabs(stiff.d[t] - (intercept + slope * t));
            if (gap > worst) worst = gap;
        }
        /* the limit is reached to O(1 / lambda) times the curvature, and the
           solve is good to (1 + 16 lambda) u, which at lambda = 1e12 is the
           larger of the two */
        CHECK(worst <= hp_bound(1e12, a.d, T, 1), "lambda 1e12: largest distance from the least squares line %g", worst);
        mat_free(stiff);
    }
    mat_free(ta); mat_free(tb); mat_free(tc); mat_free(tr);
    mat_free(a); mat_free(b); mat_free(combined); mat_free(reversed);
}

static void test_many_series(void) {
    puts("a matrix along each axis against one series at a time, a strided view, tensors, a NaN kept to its series");
    Rng rng = rng_new(7, 0);
    int T = 120, m = 9;
    Mat parent = mat_new(T + 2, m + 4);
    for (int i = 0; i < parent.r * parent.c; i++) parent.d[i] = check_non_finite(0);
    Mat y = mat_slice(parent, 1, 1 + T, 2, 2 + m);
    for (int t = 0; t < T; t++)
        for (int j = 0; j < m; j++) AT(y, t, j) = (mreal)(10 * j + rng_normal(&rng));
    Mat trends = mat_hp_trend(y, 1600, 0);
    Mat transposed = mat_T(y);
    Mat trends_rows = mat_hp_trend(transposed, 1600, 1);
    for (int j = 0; j < m; j++) {
        Mat column = mat_new(T, 1);
        for (int t = 0; t < T; t++) AT(column, t, 0) = AT(y, t, j);
        Mat alone = mat_hp_trend(column, 1600, 0);
        double bound = hp_bound(1600, column.d, T, 1);
        for (int t = 0; t < T; t++) {
            CHECK_NEAR(AT(trends, t, j), alone.d[t], bound, "column against the same series alone");
            CHECK_NEAR(AT(trends_rows, j, t), alone.d[t], bound, "row against the same series alone");
        }
        mat_free(alone); mat_free(column);
    }

    int shape[3] = { 3, T, 4 };
    Tensor t3 = tensor_new(3, shape);
    for (size_t i = 0; i < tensor_size(t3); i++) t3.d[i] = (mreal)rng_normal(&rng);
    for (int axis = 0; axis < 3; axis++) {
        Tensor trend = tensor_hp_trend(t3, 1600, axis);
        Tensor cycle = tensor_hp_cycle(t3, 1600, axis - 3);
        for (size_t i = 0; i < tensor_size(t3); i++)
            CHECK_NEAR((double)trend.d[i] + (double)cycle.d[i], t3.d[i], 4 * unit_roundoff * (1 + fabs((double)t3.d[i])), "trend + cycle");
        tensor_free(trend); tensor_free(cycle);
    }
    int order[3] = { 1, 0, 2 };
    Tensor permuted = tensor_permute(t3, order);
    Tensor from_view = tensor_hp_trend(permuted, 1600, 0), from_original = tensor_hp_trend(t3, 1600, 1);
    for (int a = 0; a < 3; a++)
        for (int t = 0; t < T; t++)
            for (int b = 0; b < 4; b++)
                CHECK(TAT3(from_view, t, a, b) == TAT3(from_original, a, t, b), "permuted view against the original");
    tensor_free(from_view); tensor_free(from_original); tensor_free(t3);

    Mat poisoned = mat_copy(y);
    AT(poisoned, T / 2, 3) = check_non_finite(0);
    Mat poisoned_trends = mat_hp_trend(poisoned, 1600, 0);
    int contained = 1, spread = 1;
    for (int t = 0; t < T; t++)
        for (int j = 0; j < m; j++) {
            if (j == 3) spread &= check_stored_non_finite(&AT(poisoned_trends, t, j), 0);
            else contained &= AT(poisoned_trends, t, j) == AT(trends, t, j);
        }
    CHECK(spread, "a NaN spreads through its own series");
    CHECK(contained, "a NaN reaches no other series");
    mat_free(poisoned_trends); mat_free(poisoned);
    mat_free(trends); mat_free(trends_rows); mat_free(transposed); mat_free(parent);
}

static void test_frame(void) {
    puts("a DataFrame's cycle and trend keep its string columns, names and row names");
    DataFrame df = df_new(6);
    Vec x = mat_lit(6, 1, 1.f, 4.f, 2.f, 8.f, 5.f, 7.f);
    const char *labels[6] = { "a", "b", "c", "d", "e", "f" };
    df_add_numeric_col(&df, "x", x);
    df_add_string_col(&df, "label", labels);
    DataFrame cycle = df_hp_cycle(&df, 1600), trend = df_hp_trend(&df, 1600);
    Mat expected = mat_hp_cycle(x, 1600, 0);
    for (int t = 0; t < 6; t++) {
        CHECK(AT(df_col_numeric(&cycle, "x"), t, 0) == expected.d[t], "frame cycle equals mat_hp_cycle");
        CHECK_NEAR((double)AT(df_col_numeric(&cycle, "x"), t, 0) + (double)AT(df_col_numeric(&trend, "x"), t, 0), x.d[t],
                   4 * unit_roundoff * 8, "cycle + trend");
    }
    CHECK(strcmp(df_col_string(&cycle, "label")[5], "f") == 0, "string column copied");
    mat_free(expected); df_free(&cycle); df_free(&trend); df_free(&df); mat_free(x);
}

int main(void) {
    test_stata();
    test_properties_and_reference();
    test_structure();
    test_many_series();
    test_frame();
    if (failures) { printf("hp_filter_correctness: %d failures\n", failures); return 1; }
    puts("hp_filter_correctness: all passed");
    return 0;
}
