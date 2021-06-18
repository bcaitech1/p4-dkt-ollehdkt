// ●는 변경한 곳（편집용）
var svgWidth = 320;	// SVG 요소의 넓이
var svgHeight = 240;	// SVG 요소의 높이
var dataSet = [50, 30, 12, 5, 3];	// 데이터셋. 비율을 나타냄
var color = d3.scale.category10();  // D3.js가 준비한 표준 10색을 지정
// 원 그래프의 좌표값을 계산하는 메서드
var pie = d3.layout.pie()	// 원 그래프 레이아웃
  .value(function(d, i){ return d; })	// 데이터셋의 데이터 반환
// 원 그래프의 안쪽 반지름, 바깥쪽 반지름 설정
var arc = d3.svg.arc().innerRadius(30).outerRadius(100)
// 원 그래프 그리기
var pieElements = d3.select("#myGraph")
  .selectAll("g")	// g 요소 지정●
  .data(pie(dataSet))	// 데이터를 요소에 연결
  .enter()
  .append("g")	// 중심 계산을 위해 그룹화하기●
  .attr("transform", "translate("+svgWidth/2+", "+svgHeight/2+")")    // 원 그래프의 중심으로 함●
// 데이터 추가
pieElements	// 데이터 수만큼 반복
  .append("path")	// 데이터의 수만큼 path 요소가 추가됨
  .attr("class", "pie")	// CSS 클래스 설정
  .style("fill", function(d, i){
		return color(i);	// 표준 10색 중 색을 반환
	})
  .transition()
  .duration(200)
  .delay(function(d,i){   // 그릴 원 그래프의 시간을 어긋나게 표시
		return i*200;
	})
  .ease("linear")	// 직선적인 움직임으로 변경
  .attrTween("d", function(d, i){	// 보간 처리
		var interpolate = d3.interpolate(
			{ startAngle : d.startAngle, endAngle : d.startAngle }, // 각 부분의 시작 각도
			{ startAngle : d.startAngle, endAngle : d.endAngle }    // 각 부분의 종료 각도
       	 );
		return function(t){
			return arc(interpolate(t)); // 시간에 따라 처리
		}
	})
// 합계와 문자 표시
var textElements = d3.select("#myGraph")
  .append("text")	// text 요소 추가
  .attr("class", "total")	// CSS 클래스 설정
  .attr("transform", "translate("+svgWidth/2+", "+(svgHeight/2+5)+")")    // 중심에 표시
  .text("合計:" + d3.sum(dataSet))	// 합계 표시
// 숫자를 부채꼴의 가운데에 표시●↓
pieElements
  .append("text")	// 데이터 수만큼 text 요소가 추가됨
  .attr("class", "pieNum")	// CSS 클래스 설정
  .attr("transform", function(d, i){
		return "translate("+arc.centroid(d)+")";    // 부채꼴의 중심으로 함(무게 중심 계산)
	})
  .text(function(d, i){
		return d.value;	// 값 표시
	})
