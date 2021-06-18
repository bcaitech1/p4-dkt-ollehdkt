var svgWidth = 320;	// SVG 요소의 넓이
var svgHeight = 240;	// SVG 요소의 높이
var offsetX = 30;	// 가로 오프셋
var offsetY = 20;	// 세로 오프셋
var scale = 2.0;	// 2배 크기로 그리기
var dataSet = [
	[
		{ year : 2004, value : 10 },
		{ year : 2005, value : 47 },
		{ year : 2006, value : 65 },
		{ year : 2007, value : 8 },
		{ year : 2008, value : 64 },
		{ year : 2009, value : 99 },
		{ year : 2010, value : 75 },
		{ year : 2011, value : 22 },
		{ year : 2012, value : 63 },
		{ year : 2013, value : 80 }
	],
	[
		{ year : 2004, value : 90 },
		{ year : 2005, value : 77 },
		{ year : 2006, value : 55 },
		{ year : 2007, value : 48 },
		{ year : 2008, value : 64 },
		{ year : 2009, value : 90 },
		{ year : 2010, value : 85 },
		{ year : 2011, value : 42 },
		{ year : 2012, value : 13 },
		{ year : 2013, value : 40 }
	],
	[
		{ year : 2004, value : 50 },
		{ year : 2005, value : 27 },
		{ year : 2006, value : 45 },
		{ year : 2007, value : 58 },
		{ year : 2008, value : 84 },
		{ year : 2009, value : 70 },
		{ year : 2010, value : 45 },
		{ year : 2011, value : 22 },
		{ year : 2012, value : 30 },
		{ year : 2013, value : 90 }
	]
];
var margin = svgWidth /(dataSet[0].length - 1);	// 꺾은선 그래프의 간격 계산
drawGraph(dataSet[0], "itemA", "linear");	// 데이터셋 안의 첫 번째 데이터
drawGraph(dataSet[1], "itemB", "linear");	// 데이터셋 안의 두 번째 데이터
drawGraph(dataSet[2], "itemC", "linear");	// 데이터셋의 세 번째 데이터
drawScale();	// 눈금 표시
// 꺾은선 그래프를 표시하는 함수
function drawGraph(dataSet, cssClassName, type){
	// 꺾은선 그래프의 좌표를 계산하는 메서드
	var line = d3.svg.line()	// svg의 선
	  .x(function(d, i){
			return offsetX + i * margin;	// X 좌표는 표시 순서×간격
		})
	  .y(function(d, i){
			return svgHeight - (d.value * scale) - offsetY;	// 데이터로부터 Y 좌표 빼기
		})
	  .interpolate(type)	// 꺾은선 그래프의 모양 설정
	// 꺾은선 그래프 그리기
	var lineElements = d3.select("#myGraph")
	  .append("path")	// 데이터 수만큼 path 요소가 추가됨
	  .attr("class", "line "+cssClassName)	// CSS 클래스 지정
	  .attr("d", line(dataSet))	//연속선 지정
}
// 그래프의 눈금을 표시하는 함수
function drawScale(){
	// 눈금을 표시하기 위해 D3 스케일 설정
	var yScale = d3.scale.linear()  // 스케일 설정
	  .domain([0, 100])   // 원래 크기
	  .range([scale*100, 0]) // 실제 표시 크기
	// 눈금 표시
	d3.select("#myGraph")	// SVG 요소를 지정
		  .append("g")	// g 요소 추가. 이것이 눈금을 표시하는 요소가 됨
		  .attr("class", "axis")	// CSS 클래스 지정
		  .attr("transform", "translate("+offsetX+", "+offsetY+")")
		  .call(
				d3.svg.axis()
			  .scale(yScale)  //스케일 적용
			  .orient("left") //눈금 표시 위치를 왼쪽으로 지정
			)
		// 가로 방향의 선을 표시
		d3.select("#myGraph")
		  .append("rect")	// rect 요소 추가
		  .attr("class", "axis_x")	// CSS 클래스 지정
		  .attr("width", svgWidth)	// 선의 굵기를 지정
		  .attr("height", 1)	// 선의 높이를 지정
		  .attr("transform", "translate("+offsetX+", "+(svgHeight-offsetY-0.5)+")")
}

