var arrVal = 0;
var chosenVal = 0;
// Shorthand for $( document ).ready()
$(function() {
	populateYear5NavItems();
	hideConceptDiv();
	//debugger;
});

function populateYear5NavItems(){
	$('#navBtns').empty();//clear before populating new items.
	$('#navigationMenu').show();
	let navBtnData = fetchJSONObject(_selectedYr);
	$.each(navBtnData,function(key, value){
		//console.log(JSON.parse(JSON.stringify(key)));
		$.each(value,function(index, json){
			//console.log(jsonBtn);
			if(json.enabled == true){
				$(json.addTo).append('<button type=\"button\" class=\"btn btn-outline-warning\" onClick=\"' + json.clickFunc + '\">' + json.name + '</button> &nbsp;');
			} else{
				$(json.addTo).append('<button type=\"button\" class=\"btn btn-outline-warning disabled\" data-bs-toggle=\"tooltip\" data-bs-placement=\"bottom\" title=\"WORK In Progress !!\">' + json.name + '</button> &nbsp;');
			}
		});
	});
	enableAllTooltips();
}

function reload(){
	//location.reload();
	resetAllParams();
}
function getCheckAnswerButton(){
    return $('#chkans');
}
function getDdMenu(){
	return $('#ddMenu');
}
function getChooseMenu(){
	return $('#choose');
}
function getLoadQuestionsButton(){
	return $('#lqst');
}
function getResultDiv(){
	return $("#resultDiv");
}
function hideConceptDiv(){
	$('#conceptDiv').hide();
}
function showConceptDiv(){
	$('#conceptDiv').show();
}

function setResult(className, resultText){
	
	let divPointer = getResultDiv();
	divPointer.prop("className", className);
	divPointer.prop("innerHTML", "<strong>" + resultText + "</strong>");
	divPointer.prop("style", "display:block");
}

function enableChooseMenu(val){
	let divPointer = getChooseMenu();
	divPointer.prop("className", "btn btn-primary dropdown-toggle");
	divPointer.prop("innerHTML", "Choose " + val + " Type :");
}

function clearDdMenu(){
	getDdMenu().empty();
	getChooseMenu().html("Choose Option:").prop("className", "btn btn-secondary dropdown-toggle disabled");
}

function enableLoadQuestionsButton(){
	getLoadQuestionsButton().prop("className", "btn btn-primary btn-sm p-3 mt-2");
}

function disableLoadQuestionsButton(){
	getLoadQuestionsButton().prop("className", "btn btn-secondary btn-sm disabled p-3 mt-2");
}

function disableChkAnsButton(){
	getCheckAnswerButton().prop("className", "btn btn-secondary btn-sm disabled p-3 mt-2");
}

function enableChkAnsButton(){
	$("#chkans").prop("className", "btn btn-primary btn-sm p-3 mt-2");
}

function clearAllQuestions(){
	$('#questions').prop("innerHTML", "");
}

function resetAllParams(){
	hideConceptDiv();
	clearDdMenu();
	reset();
	hideResult();
	clearAllQuestions();
	disableLoadQuestionsButton();
	disableChkAnsButton();
	$("#output").html("");
}
/**
 Called from
**/
function displayChoose(val){
	resetAllParams();
	let ddMenu = getDdMenu();
	for(i=0;i<getSelectionLength(val);i++){
		if(val == "Division"){
			ddMenu.append("<a class=\"dropdown-item\" onClick=\"loadDivisionParams(4,"+i+");\">"+fetchJSONObject(landingMenuLabel).divisionDisplay[i]+"</a></li>");
		} else if(val == "Addition"){
			ddMenu.append("<a class=\"dropdown-item\" onClick=\"loadAdditionParams(1,"+i+");\">"+fetchJSONObject(landingMenuLabel).commonDisplay[i]+"</a></li>");
		} else if(val == "Subtraction"){
			ddMenu.append("<a class=\"dropdown-item\" onClick=\"loadSubtractionParams(2,"+i+");\">"+fetchJSONObject(landingMenuLabel).divisionDisplay[i]+"</a></li>");
		} else if(val == "Multiplication"){
			ddMenu.append("<a class=\"dropdown-item\" onClick=\"loadMultiplicationParams(3,"+i+");\">"+fetchJSONObject(landingMenuLabel).multiplicationDisplay[i]+"</a></li>");
		}
	}
	enableChooseMenu(val);
}
/**
 * This function returns the dropdown choose menu length.
 **/
function getSelectionLength(val){
    if(val == "Division" || val == "Subtraction"){
        return fetchJSONObject(landingMenuLabel).divisionDisplay.length;
    } else if(val == "Addition"){
        return fetchJSONObject(landingMenuLabel).commonDisplay.length;
    } else if(val == "Multiplication"){
        return fetchJSONObject(landingMenuLabel).multiplicationDisplay.length;
    }
}

function loadAdditionParams(chosenVal, arrVal){
	clearAllQuestions();
	hideResult();
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
		duration = maxQ * 5;
	}
	reset();
	getChooseMenu().prop("innerHTML", "Add : " + fetchJSONObject(landingMenuLabel).commonDisplay[arrVal]);
	enableLoadQuestionsButton();
}
function loadSubtractionParams(chosenVal, arrVal){
	clearAllQuestions();
	hideResult();
	showConceptDiv();
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
			//3 Digit By 1 Digit
			num1[i] = getRandomInt(100,999);
			num2[i] = getRandomInt(1,9);
		} else if(arrVal == 4){
            //3 Digit By 2 Digit
            num1[i] = getRandomInt(100,999);
            num2[i] = getRandomInt(10,99);
        } else if(arrVal == 5){
			//3 Digit By 3 Digit
			num1[i] = getRandomInt(100,999);
			num2[i] = getRandomInt(100,num1[i]);
		}
		duration = maxQ * 15;
	}
	getChooseMenu().prop("innerHTML", "Subtract : " + fetchJSONObject(landingMenuLabel).divisionDisplay[arrVal]);
	enableLoadQuestionsButton();
}
function loadMultiplicationParams(chosenVal, arrVal){
	loadAdditionParams(chosenVal, arrVal);
	getChooseMenu().prop("innerHTML", "Multiply : " + fetchJSONObject(landingMenuLabel).multiplicationDisplay[arrVal]);
}
function loadDivisionParams(chosenVal, arrVal){
	clearAllQuestions();
	hideResult();
	reset();
	this.chosenVal = chosenVal;
	this.arrVal = arrVal;
	for (i = 0; i < maxQ; i++) {
		if(arrVal == 0){
			//1 Digit By 1 Digit
			num1[i] = getRandomInt(1,9);
			num2[i] = getRandomInt(1,num1[i]);
			duration = maxQ * 2;
			//getRandomInt(1,num1[i]);
		} else if(arrVal == 1){
			//2 Digit By 1 Digit
			num1[i] = getRandomInt(10,99);
			num2[i] = getRandomInt(1,9);
			duration = maxQ * 2;
		} else if(arrVal == 2){
			//2 Digit By 2 Digit
			num1[i] = getRandomInt(10,99);
			num2[i] = getRandomInt(10,num1[i]);
			duration = maxQ * 5;
		}  else if(arrVal == 3){
            //3 Digit By 1 Digit
            num1[i] = getRandomInt(100,999);
            num2[i] = getRandomInt(1,9);
			duration = maxQ * 7;
        } else if(arrVal == 4){
			//3 Digit By 2 Digit
			num1[i] = getRandomInt(100,999);
			num2[i] = getRandomInt(10,99);
			duration = maxQ * 10;
		} else if(arrVal == 5){
			//3 Digit By 3 Digit
			num1[i] = getRandomInt(100,999);
			num2[i] = getRandomInt(100,num1[i]);
			duration = maxQ * 15;
		}
	}
	getChooseMenu().prop("innerHTML", "Divide : " + fetchJSONObject(landingMenuLabel).divisionDisplay[arrVal]);
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
    startTimer();
	enableChkAnsButton();
}

function getRandomInt(min, max) {
    return Math.floor(Math.random() * (max - min)) + min;
}

function checkAns() {
    pauseTimer();
    disableAllYear5Questions();
    let correct = 0;
    for (i = 0; i < maxQ; i++) {
		let result = getResult(i);
		debugger;
        var elementId = i + 1;
        if ($("#Q" + elementId).val().length != 0) {
            //let value = parseInt($("#Q" + elementId).val());
            let value = $("#Q" + elementId).val();
            if (parseFloat(value).toFixed(2) == parseFloat(result).toFixed(2)) {
                $("#sp" + elementId).prop("innerHTML", "&#x2714;");
                correct = correct + 1;
            } else {
                $("#sp" + elementId).prop("innerHTML", "&#x2716;");
				$("#correctAns" + elementId).prop("innerHTML", "&nbsp;Correct Answer is: " + result);
            }
        } else {
            $("#sp" + elementId).prop("innerHTML", "&#x2718;");
			$("#correctAns" + elementId).prop("innerHTML", "<font color=\"red\">&nbsp;Correct Answer is: " + result + "</font>");
        }
    }
    if (correct == maxQ) {
        showResult(100, "Huraaaaaaayyyy! You Got 100%.");
    } else if (correct == 0) {
        showResult(0, "YIKES! You Got 0%.");
    } else {
        let result = ((correct / maxQ) * 100).toFixed(2);
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
    var x = getResultDiv().hide();
}

function disableAllYear5Questions(){
    //let olObj = document.getElementById('questions');
    //$("div *").disable();
    $('#questions :input').attr('disabled', true);
}