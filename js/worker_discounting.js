self.importScripts('random.min.js');
self.importScripts('math.min.js');

var generator;
var random;

/* Const for BIC/AIC */
var M_PI = 3.14159265358979323846;

/* DE Heuristics */ 
var populationSize = 1000;
var populationAgents = [];
var populationAgentsCost = [];

/* DE Parameters */ 
var mutationFactor = 0.8;
var crossoverRate = 0.9;

/* Comparison Holders */
var lowestRecordedCost = Number.MAX_VALUE;
var lowestLossIndex = 0;

/* Data, Settings */
var xValues, yValues;
var maxits, boundedRachlin;

/* Differential Evolution, Helpers */
function costFunctionNoise(inputs, delay) { return inputs[0]; }
function costFunctionExponential(inputs, delay) { return Math.exp(-Math.exp(inputs[0]) * delay); }
function costFunctionHyperbolic(inputs, delay) { return (1 / (1 + Math.exp(inputs[0]) * delay)); }
function costFunctionBetaDelta(inputs, delay) { return InvIt(inputs[0]) * Math.pow(InvIt(inputs[1]), delay); }
function costFunctionGreenMyerson(inputs, delay) { return 1 / Math.pow((1 + Math.exp(inputs[0]) * delay), Math.exp(inputs[1])); }
function costFunctionRachlin(inputs, delay) { return Math.pow((1 + Math.exp(inputs[0]) * Math.pow(delay, Math.exp(inputs[1]))), -1); }
function costFunctionGeneralizedHyperbolic(inputs, delay) { return Math.pow((1 + delay * Math.exp(inputs[0])),(-(Math.exp(inputs[1]) / Math.exp(inputs[0])))); }
function costFunctionEbertPrelec(inputs, delay) { return Math.exp(-Math.pow((Math.exp(inputs[0])*delay), Math.exp(inputs[1]))); }
function costFunctionBeleichrodt(inputs, delay) { return (InvIt(inputs[2]) * Math.exp(-Math.exp(inputs[0]) * Math.pow(delay, Math.exp(inputs[1])))); }

/* Shell of fx for calculating model costs */
function costFunctionShell(inputs, func)
{
    var val = 0.0;
    var tempDelay, tempValue;

    var temp;

    for (var j = 0; j < xValues.length; j++)
    {
        tempDelay = xValues[j];
        tempValue = yValues[j];

        temp = tempValue - func(inputs, tempDelay);

        val = val + (temp * temp);
    }

    val = val / parseFloat(xValues.length);

    return val;
}

/* Math.js */
function integrate(func, start, end, stepSize) 
{
  var area = 0;

  stepSize = stepSize || 0.01;

  for (var x = start; x < end; x += stepSize) 
    area += func(x + stepSize / 2) * stepSize;

  return area;
}

math.import({
   integrate: integrate
})

/* END Math.js */

/* Invert Transform */
function InvIt(value) { return Math.exp(value) / (Math.exp(value) + 1); }

/* Calculate and assign Bayes Factor */
function CalculateBF(result, noiseBic) { result.BF = Math.exp(-0.5 * (result.BIC - noiseBic)); }

/* Calculate and assign Probability */
function CalculateProbabilities(result, bayesSum) { result.Probability = result.BF / bayesSum; }

/* Calculate and assign AIC */
function CalculateAIC(result)
{
    var N = xValues.length;
    var DF = result.Params.length + 1;
    var SSR = result.MSE * N;

    result.AIC = N + N * Math.log(2 * M_PI) + N * Math.log(SSR/N) + 2 * (DF);
}

/* Calculate and assign BIC */
function CalculateBIC(result)
{
    var N = xValues.length;
    var DF = result.Params.length + 1;
    var SSR = result.MSE * N;

    result.BIC = N + N * Math.log(2 * M_PI) + N * Math.log(SSR/N) + Math.log(N) * (DF);
}

/* Re-scale wonky model parameters */
function restoreBeleichrodt(result)
{
    var replacementArray = new Array(3);

    replacementArray[0] = Math.exp(result.Params[0])
    replacementArray[1] = Math.exp(result.Params[1])
    replacementArray[2] = InvIt(result.Params[2])

    result.Params = replacementArray;
}

/* Sort by model probability */
function SortResults(a, b) 
{
  if (a.Probability < b.Probability)
    return 1;

  if (a.Probability > b.Probability)
    return -1;

  return 0;
}

/* Get best population agent */
function GetBestAgent() { return populationAgents[lowestLossIndex]; }

/* Get best population agent cost */
function GetBestCost() { return populationAgentsCost[lowestLossIndex]; }

/* Routing for MB-AUC, by model */
function CalculateAUC(obj, lowerX, upperX)
{
    var fitter = undefined;

    if (obj.Model == "Noise")
    {
        obj.AUC = parseFloat(obj.Params[0]);

        return
    }
    else if ( obj.Model == "Exponential")
        fitter = function(x) { return Math.exp(-obj.Params[0] * x); }
    else if ( obj.Model == "Hyperbolic")
        fitter = function(x) { return 1/(1 + obj.Params[0] * x); }
    else if ( obj.Model == "Beta-Delta")
        fitter = function(x) { return obj.Params[0] * Math.exp(-(1.0-obj.Params[1]) * x); }
    else if ( obj.Model == "Green-Myerson")
        fitter = function(x) { return 1/Math.pow((1 + obj.Params[0]*x), obj.Params[1]); }
    else if ( obj.Model == "Rachlin")
        fitter = function(x) { return Math.pow((1+obj.Params[0]*(Math.pow(x,obj.Params[1]))),(-1)); }
    else if ( obj.Model == "Loewstein-Prelec")
        fitter = function(x) { return Math.pow((1 + x * obj.Params[0]),(-obj.Params[1] / obj.Params[0])); }
    else if ( obj.Model == "Ebert-Prelec")
        fitter = function(x) { return Math.exp(-Math.pow((obj.Params[0]*x), obj.Params[1])); }
    else if ( obj.Model == "Beleichrodt")
        fitter = function(x) { return obj.Params[2] * Math.exp(-(obj.Params[0] * Math.pow(x,obj.Params[1]))); }

    var area = math.integrate( fitter, lowerX, upperX);
    obj.AUC = area / (upperX - lowerX);
}

/* Routing for MB-AUC, by model */
function CalculateAUClog10(obj, lowerX, upperX)
{
    var fitter = undefined;

    if ( obj.Model == "Noise")
    {
        obj.AUClog10 = parseFloat(obj.Params[0]);

        return
    }
    else if ( obj.Model == "Exponential")
        fitter = function(x) { return Math.exp(-obj.Params[0] * Math.pow(10, x)); }
    else if ( obj.Model == "Hyperbolic")
        fitter = function(x) { return 1/(1 + obj.Params[0] * Math.pow(10, x)); }
    else if ( obj.Model == "Beta-Delta")
        fitter = function(x) { return obj.Params[0] * Math.exp(-(1.0-obj.Params[1]) * Math.pow(10, x)); }
    else if ( obj.Model == "Green-Myerson")
        fitter = function(x) { return 1/Math.pow((1 + obj.Params[0]*Math.pow(10, x)), obj.Params[1]); }
    else if ( obj.Model == "Rachlin")
        fitter = function(x) { return Math.pow((1+obj.Params[0]*(Math.pow(Math.pow(10, x),obj.Params[1]))),(-1)); }
    else if ( obj.Model == "Loewstein-Prelec")
        fitter = function(x) { return Math.pow((1 + Math.pow(10, x) * obj.Params[0]),(-obj.Params[1] / obj.Params[0])); }
    else if ( obj.Model == "Ebert-Prelec")
        fitter = function(x) { return Math.exp(-Math.pow((obj.Params[0]*Math.pow(10, x)), obj.Params[1])); }
    else if ( obj.Model == "Beleichrodt")
        fitter = function(x) { return obj.Params[2] * Math.exp(-(obj.Params[0] * Math.pow(Math.pow(10, x), obj.Params[1]))); }

    if (lowerX <= 1)
    {
        lowerX = Math.log10(1);
        upperX = Math.log10((upperX + 1));
    }

    var area = math.integrate( fitter, lowerX, upperX);
    obj.AUClog10 = area / (upperX - lowerX);
}

/* Routing for ED50, by model */
function CalculateED50(obj, lowerX, upperX)
{
    var fitter = undefined;

    if (obj.Model == "Noise")
    {
        obj.ED50 = Number.NaN;

        return;
    }
    else if (obj.Model == "Exponential")
    {
        obj.ED50 = Math.log(Math.log(2)/obj.Params[0]);

        return;
    }
    else if (obj.Model == "Hyperbolic")
    {
        obj.ED50 = Math.log(1/(obj.Params[0]));

        return;
    }
    else if (obj.Model == "Beta-Delta")
    {
        obj.ED50 = Math.log(Math.log((1/(2*obj.Params[0])))/Math.log(obj.Params[1]));

        return;
    }
    else if (obj.Model == "Green-Myerson")
    {
        obj.ED50 = Math.log((Math.pow(2,(1/obj.Params[1]))-1)/obj.Params[0]);

        return;
    }
    else if (obj.Model == "Rachlin")
    {
        obj.ED50 = Math.log(Math.pow((1/(obj.Params[0])), (1/obj.Params[1])));

        return;
    }
    else if (obj.Model == "Loewstein-Prelec")
    {
        fitter = function(k, x) { return Math.pow((1 + x * k[0]),(-k[1] / k[0])); }
    }
    else if (obj.Model == "Ebert-Prelec")
    {
        fitter = function(k, x) { return Math.exp(-Math.pow((k[0] * x), k[1])); }
    }
    else if (obj.Model == "Beleichrodt")
    {
        fitter = function(k, x) { return k[2] * Math.exp(-k[0] * Math.pow(x, k[1])); }
    }

    obj.ED50 = empiricalED50Process(fitter, obj.Params, upperX);
}

/* Manual Hueristic for determining ED50 */
function empiricalED50Process(fitFunction, params, upperX) 
{
    var lowDelay = 0;
    var highDelay = upperX * 100;
    var i = 0;

    while ( (highDelay - lowDelay) > 0.001 && i < 100) 
    {
        var lowEst  = fitFunction( params, lowDelay);
        var midEst  = fitFunction( params, (lowDelay + highDelay) / 2);
        var highEst = fitFunction( params, highDelay);

        if ( lowEst > 0.5 && midEst > 0.5) 
        {
            //Above 50% mark range
            lowDelay  = (lowDelay+highDelay)/2;
            highDelay = highDelay;

        } 
        else if ( highEst < 0.5 && midEst < 0.5) 
        {
            //Below 50% mark range
            lowDelay  = lowDelay;
            highDelay = (lowDelay+highDelay)/2;
        }

        i++;
    }

    return Math.log((lowDelay+highDelay)/2);
}

/* DE Methods */
function Optimize(costFunc, setLower = -100, setUpper = 100, parameterCount = 1)
{
    populationAgents = new Array(populationSize);
    populationAgentsCost = new Array(populationSize);

    for ( var i = 0; i < populationSize; i++)
    {
        populationAgents[i] = new Array(parameterCount);

        for ( var j = 0; j < parameterCount; j++)
            populationAgents[i][j] = random.real(setLower, setUpper);
    }

    // Process initial costs
    for ( var i = 0; i < populationSize; i++)
    {
        populationAgentsCost[i] = costFunctionShell(populationAgents[i], costFunc);

        if (populationAgentsCost[i] < lowestRecordedCost)
        {
            lowestRecordedCost = populationAgentsCost[i];
            lowestLossIndex = i;
        }
    }

    for ( var i = 0; i < maxits; i++)
        SelectionAndCrossing(costFunc, parameterCount, setLower, setUpper);
}

function SelectionAndCrossing(costFunc, nParams, boundLower, boundUpper)
{
    var minCost = populationAgentsCost[0];
    var bestAgentIndex = 0;
    // no need to make new dist

    for (var x = 0; x < populationSize; x++)
    {
        var a = x;
        var b = x;
        var c = x;

        // Agents must be different from each other and from x
        while (a == x || b == x || c == x || a == b || a == c || b == c)
        {
            a = random.integer(0, populationSize - 1);
            b = random.integer(0, populationSize - 1);
            c = random.integer(0, populationSize - 1);
        }

        var z = new Array(nParams);
        for (var i = 0; i < nParams; i ++)
            z[i] = populationAgents[a][i] + mutationFactor * (populationAgents[b][i] - populationAgents[c][i]);

        var R = random.integer(0, nParams - 1);

        var r = new Array(nParams);

        for (var i = 0; i < nParams; i++)
            r[i] = random.real(0, 1);

        var newX = new Array(nParams);

        // Execute crossing
        for (var i = 0; i < nParams; i++)
        {
            if (r[i] < crossoverRate || i == R)
                newX[i] = z[i];
            else
                newX[i] = populationAgents[x][i];
        }

        // Check if newX candidate satisfies constraints and skip it if not.
        // If agent is skipped loop iteration x is decreased so that it is ensured
        // that the population has constant size (equal to populationAgentsSize).
        if (newX > boundUpper || newX < boundLower)
        {
            x--;

            continue;
        }

        var newCost = costFunctionShell(newX, costFunc);

        if (newCost < populationAgentsCost[x])
        {
            populationAgents[x] = newX;
            populationAgentsCost[x] = newCost;
        }

        if (populationAgentsCost[x] < minCost)
        {
            minCost = populationAgentsCost[x];
            bestAgentIndex = x;
        }
    }

    lowestRecordedCost = minCost;
    lowestLossIndex = bestAgentIndex;
}

function beginLooper()
{
    var returnArray = [];

    // Noise Model
    Optimize(costFunctionNoise, 
             setLower = 0, 
             setUpper = 1, 
             parameterCount = 1);
        var noiseResult = {
            Model: "Noise",
            Params: [parseFloat(GetBestAgent())],
            MSE: GetBestCost(),
            RMSE: Math.sqrt(GetBestCost()),
            done: false
        };

        returnArray.push(noiseResult);

        postMessage({ done: false, msg: "Calculating Exponential" });

    // Exponential Model
    Optimize(costFunctionExponential, 
             setLower = -50, 
             setUpper = 50, 
             parameterCount = 1);
        var expResult = {
            Model: "Exponential",
            Params: [Math.exp(GetBestAgent())],
            MSE: GetBestCost(),
            RMSE: Math.sqrt(GetBestCost()),
            done: false
        };

        returnArray.push(expResult);

        postMessage({ done: false, msg: "Calculating Hyperbolic" });

    // Hyperbolic Model
    Optimize(costFunctionHyperbolic, 
             setLower = -50, 
             setUpper = 50, 
             parameterCount = 1);
        var hypResult = {
            Model: "Hyperbolic",
            Params: [Math.exp(GetBestAgent())],
            MSE: GetBestCost(),
            RMSE: Math.sqrt(GetBestCost()),
            done: false
        };

        returnArray.push(hypResult);

        postMessage({ done: false, msg: "Calculating Beta-Delta" });

    // Beta-Delta Model
    Optimize(costFunctionBetaDelta, 
             setLower = -1000, 
             setUpper = 500, 
             parameterCount = 2);
        var bdResult = {
            Model: "Beta-Delta",
            Params: GetBestAgent().map(function(obj) { return InvIt(obj) }),
            MSE: GetBestCost(),
            RMSE: Math.sqrt(GetBestCost()),
            done: false
        };

        returnArray.push(bdResult);

        postMessage({ done: false, msg: "Calculating Green-Myerson" });

    // Green-Myerson Model
    Optimize(costFunctionGreenMyerson, 
             setLower = -50, 
             setUpper = 50, 
             parameterCount = 2);
        var gmResult = {
            Model: "Green-Myerson",
            Params: GetBestAgent().map(function(obj) { return Math.exp(obj) }),
            MSE: GetBestCost(),
            RMSE: Math.sqrt(GetBestCost()),
            done: false
        };

        returnArray.push(gmResult);

        postMessage({ done: false, msg: "Calculating Rachlin" });

    // Rachlin Model
    Optimize(costFunctionRachlin, 
             setLower = -50, 
             setUpper = 50, 
             parameterCount = 2);
        var rachResult = {
            Model: "Rachlin",
            Params: GetBestAgent().map(function(obj) { return Math.exp(obj) }),
            MSE: GetBestCost(),
            RMSE: Math.sqrt(GetBestCost()),
            done: false
        };

    if (!boundedRachlin || (boundedRachlin && (rachResult.Params[1] <= 1 && rachResult.Params[1] >= 0)))
        returnArray.push(rachResult);

    postMessage({ done: false, msg: "Calculating Loewstein-Prelec" });

    // Ebert-Prelec
    Optimize(costFunctionGeneralizedHyperbolic, 
             setLower = -100, 
             setUpper = 100, 
             parameterCount = 2);
        var lpResult = {
            Model: "Loewstein-Prelec",
            Params: GetBestAgent().map(function(obj) { return Math.exp(obj) }),
            MSE: GetBestCost(),
            RMSE: Math.sqrt(GetBestCost()),
            done: false
        };

        returnArray.push(lpResult);

    postMessage({ done: false, msg: "Calculating Ebert-Prelec" });

    // Ebert-Prelec
    Optimize(costFunctionEbertPrelec, 
             setLower = -10, 
             setUpper = 10, 
             parameterCount = 2);
        var epResult = {
            Model: "Ebert-Prelec",
            Params: GetBestAgent().map(function(obj) { return Math.exp(obj) }),
            MSE: GetBestCost(),
            RMSE: Math.sqrt(GetBestCost()),
            done: false
        };

        returnArray.push(epResult);

    postMessage({ done: false, msg: "Calculating Beleichrodt et al." });

    // Ebert-Prelec
    Optimize(costFunctionBeleichrodt, 
             setLower = -100, 
             setUpper = 100, 
             parameterCount = 3);
        var belResult = {
            Model: "Beleichrodt",
            Params: GetBestAgent(),
            MSE: GetBestCost(),
            RMSE: Math.sqrt(GetBestCost()),
            done: false
        };

        restoreBeleichrodt(belResult)

        returnArray.push(belResult);

        postMessage({ done: false, msg: "Calculating Supplemental Measures" });

    // Information Criteria
    for (var i = 0; i < returnArray.length; i++)
    {
        CalculateAIC(returnArray[i]);
        CalculateBIC(returnArray[i]);
    }

    var bfSum = 0;

    // Bayes factors
    for (var i = 0; i < returnArray.length; i++)
    {
        CalculateBF(returnArray[i], returnArray[0].BIC);

        bfSum = bfSum + returnArray[i].BF;
    }

    // Model probabilities
    for (var i = 0; i < returnArray.length; i++)
    {
        CalculateProbabilities(returnArray[i], bfSum);
    }

    returnArray.sort(SortResults);  

    var loX = Math.min(...xValues.map(Number))
    var hiX = Math.max(...xValues.map(Number))

    CalculateAUC(returnArray[0], loX, hiX);
    CalculateAUClog10(returnArray[0], loX, hiX);
    CalculateED50(returnArray[0], loX, hiX);


    // Fire completed event
    postMessage({
        done: true,
        results: returnArray,
        x: xValues.map(Number),
        y: yValues.map(Number)
    });
}

onmessage = function(passer)
{
    generator      = Random.engines.mt19937().seed(123);
    random         = new Random(generator);
    maxits         = passer.data.maxIterations;
    xValues        = passer.data.x;
    yValues        = passer.data.y;
    boundedRachlin = passer.data.boundRachlin;

    postMessage({ done: false, msg: "Calculating Noise" });

    beginLooper();
}