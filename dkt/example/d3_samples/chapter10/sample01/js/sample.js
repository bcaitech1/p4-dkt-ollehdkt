var svgWidth = 320;	// SVG 요소의 넓이
var svgHeight = 240;	// SVG 요소의 높이
// 데이터셋
var dataSet = [
		[30, 40], [120, 115], [125, 90], [150, 160], [300, 190],
		[60, 40], [140, 145], [165, 110], [200, 170], [250, 190]
	];
// 산포도 그리기
var circleElements = d3.select("#myGraph")
  .selectAll("circle")
  .data(dataSet)
  .enter()
  .append("circle")	// 데이터의 개수만큼 circle 요소가 추가됨
  .attr("class", "mark")	// CSS 클래스 지정
  .attr("cx", function(d, i){
		return d[0];	// 최초 요소를 X 좌표로 함
	})
  .attr("cy", function(d, i){
		return svgHeight-d[1];	// 2번째의 요소를 Y 좌표로 함
	})
  .attr("r", 5)	// 반지름을 지정
