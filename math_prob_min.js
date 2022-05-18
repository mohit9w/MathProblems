let url="MenuItems.json";let landingMenuLabel="landingMenu";var maxQ=10;var num1=Array(maxQ).fill(0);var num2=Array(maxQ).fill(0);var time=0;var running=0;var _selectedYr;var _AllLoadedYrs=[];function resetAll(){$('#chooseSub').prop("className","btn btn-outline-warning dropdown-toggle disabled");$('#chooseSub').html("Choose Subject:");$('#subMenu').empty();$('#yearData').empty();$('#navBtns').empty();$('#navigationMenu').hide();}
$(function(){addProjectEventListeners();loadLandingMenuData();enableAllTooltips();$('#navigationMenu').hide();Storage.prototype.setJSON=function(key,obj){return this.setItem(key,JSON.stringify(obj))}
Storage.prototype.getJSON=function(key){return JSON.parse(this.getItem(key))}});function loadLandingMenuData(){$.getJSON(url,function(data){storeJSONObject(landingMenuLabel,data);loadChooseYrMenu();});}
function enableAllTooltips(){var tooltipTriggerList=[].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
var tooltipList=tooltipTriggerList.map(function(tooltipTriggerEl){return new bootstrap.Tooltip(tooltipTriggerEl)});$('[data-bs-toggle="tooltip"]').prop("style","pointer-events:auto");}
function loadChooseYrMenu(){$.each(fetchJSONObject(landingMenuLabel).years,function(index,name){$('#yrMenu').append('<li class=\"btn-outline-warning\"><a class=\"dropdown-item btn-outline-warning\" onClick=\"saveYr(\''+name+'\')\">'+name+'</a></li>');});}
function saveYr(year){if(_selectedYr!=year){resetAll();_selectedYr=year.replace(" ","");$('#chooseYr').prop("innerHTML",year);populateSubjectMenu();}}
function populateSubjectMenu(){if(fetchJSONObject(_selectedYr)==null||Object.keys(fetchJSONObject(_selectedYr)).length==0){let jsonFile="/"+_selectedYr+"/"+_selectedYr+".json";$.getJSON(jsonFile,function(data){storeJSONObject(_selectedYr,data);loadSubjectMenu();});}else{loadSubjectMenu();}}
function loadSubjectMenu(){let classNameUpdated=0;$.each(Object.keys(fetchJSONObject(_selectedYr)),function(index,name){if(name!=null||name!='undefined'){$('#subMenu').append('<li class=\"btn-outline-warning\"><a class=\"dropdown-item btn-outline-warning\" onClick=\"loadYearAndSubject(\''+name+'\')\">'+name+'</a></li>');if(classNameUpdated==0){$('#chooseSub').prop("className","btn btn-outline-warning dropdown-toggle");classNameUpdated=1;}}});}
function loadYearAndSubject(subject){$('#chooseSub').prop("innerHTML",subject);let filePath="/"+_selectedYr+"/";let htmlFile=_selectedYr+".html";let jsFile=_selectedYr+".js";let jsonFile=_selectedYr+".json";$.get(filePath+htmlFile,function(data,status){$('#yearData').prop("innerHTML",data);});if($.inArray(_selectedYr,_AllLoadedYrs)==-1){$.getScript(filePath+jsFile,function(){});_AllLoadedYrs.push(_selectedYr);}else{let funcName='populate'+_selectedYr+'NavItems';executeFunction(funcName);}}
function clearStoredData(){window.localStorage.clear();window.sessionStorage.clear();}
function storeJSONObject(key,jsonToStore){window.localStorage.setJSON(key,jsonToStore);}
function fetchJSONObject(key){return window.localStorage.getJSON(key);}
function addProjectEventListeners(){document.addEventListener('beforeunload',clearStoredData());}
function startPause(){if(running==0){running=1;increment();}else{running=0;}}
function reset(){running=0;time=0;$("#output").html("00:00:00");}
function increment(){if(running==1){setTimeout(function(){time++;var mins=Math.floor(time/10/60);var secs=Math.floor(time/10%60);var tenths=time%10;if(mins<10){mins="0"+mins;}
if(secs<10){secs="0"+secs;}
$("#output").prop("innerHTML","<i class=\"fa fa-clock-o\"></i> "+mins+":"+secs+":"+"0"+tenths);increment();},100);}}
function executeFunction(_functionName){var fn=window[_functionName];if(typeof fn==="function"){fn();}}