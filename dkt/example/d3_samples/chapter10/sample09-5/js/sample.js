var svgWidth = 320;	// SVG 요소의 넓이
var svgHeight = 240;	// SVG 요소의 높이
// 데이터셋
var dataSet = [
		[30, 40], [120, 115], [125, 90], [150, 160], [300, 190],
		[60, 40], [140, 145], [165, 110], [200, 170], [250, 190]
	];
// 산포도 그리기
var circleElements = d3.select("#myGraph")
	.selectAll("path")
	.data(dataSet)
	.enter()
	.append("path")	// 데이터의 수만큼 path 요소가 추가됨
	.attr("transform", function(d) { return "translate(" + d[0] + "," + d[1] + ")"; })  // 위치 지정
	.attr("d", d3.svg.symbol().type("cross"))	// ▼을 지정
