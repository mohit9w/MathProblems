var maxQ = 10;
var num1 = Array(maxQ).fill(0);
var num2 = Array(maxQ).fill(0);
var commonDisplay = ["Single Digit","1 Digit With 2 Digit","2 Digit","2 Digit With 3 Digit","3 Digit"];
var divisionDisplay = ["Single Digit","2 Digit By 1 Digit","2 Digit","3 Digit By 2 Digit","3 Digit"];
var time = 0;
var running = 0;
var chosenVal = 0;
var arrVal = 0;

function reload(){
	location.reload();
}
function getDdMenu(){
	return document.getElementById('ddMenu');
}
function getChooseMenu(){
	return document.getElementById('choose');
}
function getLoadQuestionsButton(){
	return document.getElementById('lqst');
}
function getResultDiv(){
	return document.getElementById("resultDiv");
}
function setResult(className, resultText){
	getResultDiv().className = className;
	getResultDiv().innerHTML = "<strong>" + resultText + "</strong>";
	getResultDiv().style.display = "block";
}
function enableChooseMenu(val){
	getChooseMenu().innerHTML = "Choose " + val + " Type :";
	getChooseMenu().className = "btn btn-primary dropdown-toggle";
}
function clearDdMenu(){
	getDdMenu().innerHTML = "";
}
function enableLoadQuestionsButton(){
	getLoadQuestionsButton().className = "btn btn-primary btn-sm p-3 mt-2";
}
function disableLoadQuestionsButton(){
	getLoadQuestionsButton().className = "btn btn-primary btn-sm disabled p-3 mt-2";
}
function disableChkAnsButton(){
	document.getElementById("chkans").className = "btn btn-secondary btn-sm disabled p-3 mt-2";
}
function enableChkAnsButton(){
	document.getElementById("chkans").className = "btn btn-primary btn-sm p-3 mt-2";
}
function clearAllQuestions(){
	document.getElementById('questions').innerHTML = "";
}

function resetAllParams(){
	clearDdMenu();
	reset();
	hideResult();
	clearAllQuestions();
}

function displayChoose(val){
	resetAllParams();
	let ddMenu = getDdMenu();
	for(i=0;i<5;i++){
		let element = document.createElement("li");
		let aText = "";
		if(val == "Division"){
			aText = "<a class=\"dropdown-item\" onClick=\"loadDivisionParams(4,"+i+");\">"+divisionDisplay[i]+"</a></li>";
		} else if(val == "Addition"){
			aText = "<a class=\"dropdown-item\" onClick=\"loadAdditionParams(1,"+i+");\">"+commonDisplay[i]+"</a></li>";
		} else if(val == "Subtraction"){
			aText = "<a class=\"dropdown-item\" onClick=\"loadSubtractionParams(2,"+i+");\">"+divisionDisplay[i]+"</a></li>";
		} else if(val == "Multiplication"){
			aText = "<a class=\"dropdown-item\" onClick=\"loadMultiplicationParams(3,"+i+");\">"+commonDisplay[i]+"</a></li>";
		}
		
		element.innerHTML = aText;
		ddMenu.appendChild(element);
		//$('#your-div-id').load('your-html-page.html');
	}
	
	enableChooseMenu(val);
}

function loadAdditionParams(chosenVal, arrVal){
	this.chosenVal = chosenVal;
	this.arrVal = arrVal;
	for (i = 0; i < maxQ; i++) {
		if(arrVal == 0){
			//1 Digit By 1 Digit
			num1[i] = getRandomInt(1,9);
			num2[i] = getRandomInt(1,9);
		} else if(arrVal == 1){
			//1 Digit By 2 Digit
			num1[i] = getRandomInt(1,9);
			num2[i] = getRandomInt(10,99);
		} else if(arrVal == 2){
			//2 Digit By 2 Digit
			num1[i] = getRandomInt(10,99);
			num2[i] = getRandomInt(10,99);
		} else if(arrVal == 3){
			//2 Digit By 3 Digit
			num1[i] = getRandomInt(10,99);
			num2[i] = getRandomInt(100,999);
		} else if(arrVal == 4){
			//3 Digit By 3 Digit
			num1[i] = getRandomInt(100,999);
			num2[i] = getRandomInt(100,999);
		}
	}
	reset();
	getChooseMenu().innerHTML = "Add : " + commonDisplay[arrVal];
	enableLoadQuestionsButton();
}
function loadSubtractionParams(chosenVal, arrVal){
	reset();
	this.chosenVal = chosenVal;
	this.arrVal = arrVal;
	for (i = 0; i < maxQ; i++) {
		if(arrVal == 0){
			//1 Digit By 1 Digit
			num1[i] = getRandomInt(1,9);
			num2[i] = getRandomInt(1,num1[i]);
		} else if(arrVal == 1){
			//2 Digit By 1 Digit
			num1[i] = getRandomInt(10,99);
			num2[i] = getRandomInt(1,9);
		} else if(arrVal == 2){
			//2 Digit By 2 Digit
			num1[i] = getRandomInt(10,99);
			num2[i] = getRandomInt(10,num1[i]);
		} else if(arrVal == 3){
			//3 Digit By 2 Digit
			num1[i] = getRandomInt(100,999);
			num2[i] = getRandomInt(10,99);
		} else if(arrVal == 4){
			//3 Digit By 3 Digit
			num1[i] = getRandomInt(100,999);
			num2[i] = getRandomInt(100,num1[i]);
		}
	}
	getChooseMenu().innerHTML = "Subtract : " + divisionDisplay[arrVal];
	enableLoadQuestionsButton();
}
function loadMultiplicationParams(chosenVal, arrVal){
	loadAdditionParams(chosenVal, arrVal);
	getChooseMenu().innerHTML = "Multiply : " + divisionDisplay[arrVal];
}
function loadDivisionParams(chosenVal, arrVal){
	reset();
	this.chosenVal = chosenVal;
	this.arrVal = arrVal;
	for (i = 0; i < maxQ; i++) {
		if(arrVal == 0){
			//1 Digit By 1 Digit
			num1[i] = getRandomInt(1,9);
			num2[i] = getRandomInt(1,num1[i]);
			//getRandomInt(1,num1[i]);
		} else if(arrVal == 1){
			//2 Digit By 1 Digit
			num1[i] = getRandomInt(10,99);
			num2[i] = getRandomInt(1,9);
		} else if(arrVal == 2){
			//2 Digit By 2 Digit
			num1[i] = getRandomInt(10,99);
			num2[i] = getRandomInt(10,num1[i]);
		} else if(arrVal == 3){
			//3 Digit By 2 Digit
			num1[i] = getRandomInt(100,999);
			num2[i] = getRandomInt(10,99);
		} else if(arrVal == 4){
			//3 Digit By 3 Digit
			num1[i] = getRandomInt(100,999);
			num2[i] = getRandomInt(100,num1[i]);
		}
	}
	getChooseMenu().innerHTML = "Divide : " + divisionDisplay[arrVal];
	enableLoadQuestionsButton();
}

function loadData() {
	disableLoadQuestionsButton();
	hideResult();
    let olObj = document.getElementById('questions');
	olObj.innerHTML = "";
	let displaySign = " X ";
	if(chosenVal == 1){
		displaySign = " + ";
	} else if(chosenVal == 2){
		displaySign = " - ";
	} if(chosenVal == 4){
		displaySign = " / ";
	}
		
    for (i = 0; i < maxQ; i++) {
        let theId = i + 1;
        let elem = document.createElement("li");
        let liText = "<label for=\"Q" + theId + "\">" + num1[i] + displaySign + num2[i] + " =</label> <input type=\"number\" id=\"Q" + theId + "\" name=\"Q" + theId + "\">&nbsp;&nbsp;<span id = \"sp" + theId + "\"style=\"font-family: wingdings; font-size: 120%;\"></span><span id=\"correctAns"+theId+"\"></span><br><br>";
        elem.innerHTML = liText;
        olObj.appendChild(elem);
    }
    startPause();
	enableChkAnsButton();
}

function getRandomInt(min, max) {
    return Math.floor(Math.random() * (max - min)) + min;
}

function checkAns() {
    startPause();
    let correct = 0;
    for (i = 0; i < maxQ; i++) {
		let result = getResult(i);
        var elementId = i + 1;
        if (document.getElementById("Q" + elementId).value.length != 0) {
            let value = parseInt(document.getElementById("Q" + elementId).value);
            if (value == result) {
                document.getElementById("sp" + elementId).innerHTML = "&#x2714;";
                correct = correct + 1;
            } else {
                document.getElementById("sp" + elementId).innerHTML = "&#x2716;";
				document.getElementById("correctAns" + elementId).innerHTML = "&nbsp;Correct Answer is: " + result;
            }
        } else {
            document.getElementById("sp" + elementId).innerHTML = "&#x2718;";
			document.getElementById("correctAns" + elementId).innerHTML = "<font color=\"red\">&nbsp;Correct Answer is: " + result + "</font>";
        }
    }
    if (correct == maxQ) {
        showResult(100, "Huraaaaaaayyyy! You Got 100%.");
    } else if (correct == 0) {
        showResult(0, "YIKES! You Got 0%.");
    } else {
        let result = (correct / maxQ) * 100;
        showResult(result, "You Got " + result + "%.");
    }
	disableChkAnsButton();
}

function getResult(i){
	if(chosenVal == 1){
		return (num1[i] + num2[i]);
	} else if(chosenVal == 2){
		return (num1[i] - num2[i]);
	} else if(chosenVal == 3){
		return (num1[i] * num2[i]);
	} else if(chosenVal == 4){
		if((num1[i] % num2[i]) == 0) {
			return (num1[i] / num2[i]);
		} else {
			return (num1[i] / num2[i]).toFixed(2);
		}
	}
}



function showResult(res, resultText) {
	if(res == 100){
		setResult("alert alert-success", resultText);
	} else if(res < 50){
		setResult("alert alert-danger", resultText);
	} else{
		setResult("alert alert-info", resultText);
	}
}
function hideResult() {
    var x = getResultDiv().style.display = "none";
}

function startPause() {
    if (running == 0) {
        running = 1;
        increment();
    } else {
        running = 0;
    }
}

function reset() {
    running = 0;
    time = 0;
    document.getElementById("output").innerHTML = "00:00:00";
}

function increment() {
    if (running == 1) {
        setTimeout(function () {
            time++;
            var mins = Math.floor(time / 10 / 60);
            var secs = Math.floor(time / 10 % 60);
            var tenths = time % 10;
            if (mins < 10) {
                mins = "0" + mins;
            }
            if (secs < 10) {
                secs = "0" + secs;
            }
            document.getElementById("output").innerHTML = "<i class=\"fa fa-clock-o\"></i> " + mins + ":" + secs + ":" + "0" + tenths;
            increment();
        }, 100);
    }
}
