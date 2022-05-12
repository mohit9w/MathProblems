let url = "MenuItems.json";
var maxQ = 10;
var num1 = Array(maxQ).fill(0);
var num2 = Array(maxQ).fill(0);
var time = 0;
var running = 0;
var chosenVal = 0;
var arrVal = 0;
let _data;
var _selectedYr;
var _AllLoadedYrs = [];
var jsonMap = {};

// Shorthand for $( document ).ready()
$(function() {
    //console.log( $('[data-bs-toggle="tooltip"]') );
	loadLandingMenuData();
	enableAllTooltips();
});

function loadLandingMenuData(){
	$.getJSON(url, function (data) {
		_data = data;
		loadChooseYrMenu();
    });
}

function enableAllTooltips(){
	var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
	var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {  return new bootstrap.Tooltip(tooltipTriggerEl) });	
	$('[data-bs-toggle="tooltip"]').prop("style", "pointer-events:auto");
}

function loadChooseYrMenu(){
	$.each(_data.years,function(index, name){
		$('#yrMenu').append('<li class=\"btn-outline-warning\"><a class=\"dropdown-item btn-outline-warning\" onClick=\"saveYr(\''+name+'\')\">'+name+'</a></li>');
	});
}

function saveYr(year){
	_selectedYr = year.replace(" ", "");
	$('#chooseYr').prop("innerHTML", year);
	populateSubjectMenu();
}
/**
   Load selected year's JSON File
*/
function populateSubjectMenu(){
	let filePath = "/" + _selectedYr + "/";
	let jsonFile = _selectedYr + ".json";
	if(jsonMap[_selectedYr] == null || jsonMap[_selectedYr] == 'undefined'){
		$.getJSON(filePath + jsonFile, function (data) {
			jsonMap[_selectedYr] = data;
			loadSubjectMenu();
		});
	}
}

function loadSubjectMenu(){
	$.each(Object.keys(jsonMap[_selectedYr]),function(index, name){
		$('#subMenu').append('<li class=\"btn-outline-warning\"><a class=\"dropdown-item btn-outline-warning\" onClick=\"loadYearAndSubject(\''+name+'\')\">'+name+'</a></li>');
	});
	$('#chooseSub').prop("className", "btn btn-outline-warning dropdown-toggle");
}

function loadYearAndSubject(subject){
	$('#chooseSub').prop("innerHTML", subject);
	unloadLastYearFiles();
	let filePath = "/" + _selectedYr + "/";
	let htmlFile = _selectedYr + ".html";
	let jsFile = _selectedYr + ".js";
	let jsonFile = _selectedYr + ".json";
	//Load HTML
	$.get(filePath + htmlFile, function(data, status){
		$('#yearData').html(data);
		//alert("Data: " + data + "\nStatus: " + status);
	});
	//Do not Load JS file again
	//console.log('Index : ' + $.inArray(_selectedYr, _AllLoadedYrs));
	if($.inArray(_selectedYr, _AllLoadedYrs) == -1 ){
		// Load js file
		$.getScript(filePath + jsFile, function() {
			//console.log(filePath + jsFile + ' Javascript is loaded.');
		});
		_AllLoadedYrs.push(_selectedYr);
	}
}

function unloadLastYearFiles(){
	if (_AllLoadedYrs.length > 0){
		
	}
}