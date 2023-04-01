let url="MenuItems.json";let landingMenuLabel="landingMenu";var maxQ=30;var num1=Array(maxQ).fill(0);var num2=Array(maxQ).fill(0);var intervalId;var duration=null;var remainingTime=null;var isPaused=false;var _selectedYr;var _AllLoadedYrs=[];let devModeOn=false;let baseJsUrl="https://cdn.jsdelivr.net/gh/mohit9w/MathProblems@main";function resetAll(){$('#chooseSub').prop("className","btn btn-outline-warning dropdown-toggle disabled");$('#chooseSub').html("Choose Subject:");$('#subMenu').empty();$('#yearData').empty();$('#navBtns').empty();$('#navigationMenu').hide();}
$(function(){checkDevMode();addProjectEventListeners();loadLandingMenuData();enableAllTooltips();$('#navigationMenu').hide();Storage.prototype.setJSON=function(key,obj){return this.setItem(key,JSON.stringify(obj))}
Storage.prototype.getJSON=function(key){return JSON.parse(this.getItem(key))}});function checkDevMode(){let host=window.location.hostname;(host=="127.0.0.1"||host=="localhost")?devModeOn=true:devModeOn=false;}
function loadLandingMenuData(){$.getJSON(url,function(data){storeJSONObject(landingMenuLabel,data);loadChooseYrMenu();});}
function enableAllTooltips(){var tooltipTriggerList=[].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
var tooltipList=tooltipTriggerList.map(function(tooltipTriggerEl){return new bootstrap.Tooltip(tooltipTriggerEl)});$('[data-bs-toggle="tooltip"]').prop("style","pointer-events:auto");}
function loadChooseYrMenu(){$.each(fetchJSONObject(landingMenuLabel).years,function(index,name){$('#yrMenu').append('<li class=\"btn-outline-warning\"><a class=\"dropdown-item btn-outline-warning\" onClick=\"saveYr(\''+name+'\')\">'+name+'</a></li>');});}
function saveYr(year){if(_selectedYr!=year){resetAll();_selectedYr=year.replace(" ","");$('#chooseYr').prop("innerHTML",year);populateSubjectMenu();}}
function populateSubjectMenu(){if(fetchJSONObject(_selectedYr)==null||Object.keys(fetchJSONObject(_selectedYr)).length==0){let jsonFile="/"+_selectedYr+"/"+_selectedYr+".json";$.getJSON(jsonFile,function(data){storeJSONObject(_selectedYr,data);loadSubjectMenu();});}else{loadSubjectMenu();}}
function loadSubjectMenu(){let classNameUpdated=0;$.each(Object.keys(fetchJSONObject(_selectedYr)),function(index,name){if(name!=null||name!='undefined'){$('#subMenu').append('<li class=\"btn-outline-warning\"><a class=\"dropdown-item btn-outline-warning\" onClick=\"loadYearAndSubject(\''+name+'\')\">'+name+'</a></li>');if(classNameUpdated==0){$('#chooseSub').prop("className","btn btn-outline-warning dropdown-toggle");classNameUpdated=1;}}});}
function loadYearAndSubject(subject){$('#chooseSub').prop("innerHTML",subject);let filePath=(devModeOn?"":baseJsUrl)+("/"+_selectedYr+"/");let htmlFile=_selectedYr+".html";let jsFile=devModeOn?(_selectedYr+".js"):(_selectedYr+"_min.js");let jsonFile=_selectedYr+".json";$.get(filePath+htmlFile,function(data,status){$('#yearData').prop("innerHTML",data);});if($.inArray(_selectedYr,_AllLoadedYrs)==-1){$.getScript(filePath+jsFile,function(){});_AllLoadedYrs.push(_selectedYr);}else{let funcName='populate'+_selectedYr+'NavItems';executeFunction(funcName);}}
function clearStoredData(){window.localStorage.clear();window.sessionStorage.clear();}
function storeJSONObject(key,jsonToStore){window.localStorage.setJSON(key,jsonToStore);}
function fetchJSONObject(key){return window.localStorage.getJSON(key);}
function addProjectEventListeners(){document.addEventListener('beforeunload',clearStoredData());}
function startPause(){if(running==0){running=1;increment();}else{running=0;}}
function startTimer(){var start=new Date();remainingTime=duration;var end=new Date(start.getTime()+remainingTime*1000);intervalId=setInterval(function(){if(!isPaused){var now=new Date();var seconds=Math.round((end.getTime()-now.getTime())/1000);if(seconds<0){alert('Time is up!');stopTimer();return;}
var hours=Math.floor(seconds/3600);seconds-=hours*3600;var minutes=Math.floor(seconds/60);seconds-=minutes*60;$('#timer').text(formatTime(hours)+':'+formatTime(minutes)+':'+formatTime(seconds));remainingTime=seconds;}},1000);}
function reset(){clearInterval(intervalId);intervalId=0;remainingTime=null;isPaused=false;$('#timer').text('00:00:00');}
function pauseTimer(){isPaused=true;}
function formatTime(time){if(time<10){return'0'+time;}else{return time;}}
function executeFunction(_functionName){var fn=window[_functionName];if(typeof fn==="function"){fn();}}