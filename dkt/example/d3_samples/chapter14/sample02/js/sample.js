var svgWidth = 160;	// SVG요소의 넓이
var svgHeight = 240;	// SVG요소의 높이
var blockSize = 20;	// 블록 크기
// 데이터셋
var dataSet = [
	0, 1, 2, 3, 3, 4, 5, 4,
	0, 0, 0, 3, 4, 4, 5, 3,
	1, 0, 0, 0, 0, 0, 0, 0,
	2, 6, 8, 7, 0, 0, 0, 2,
	4, 8, 9, 8, 0, 0, 1, 0,
	2, 6, 8, 6, 4, 0, 0, 0,
	2, 5, 3, 0, 2, 0, 0, 0,
	1, 2, 0, 0, 0, 0, 0, 0,
	1, 0, 0, 0, 0, 7, 8, 9,
	0, 0, 0, 0, 7, 9, 9, 9,
	0, 0, 0, 7, 8, 8, 9, 7,
	0, 0, 0, 6, 6, 7, 6, 5
];
// 히트맵에 표시할 색을 자동으로 계산
var color = d3.interpolateHsl("blue", "yellow");	// 파란색에서 노란색으로 보간
var maxValue = d3.max(dataSet);	// 최댓값을 구함
// 히트맵 준비
var heatMap = d3.select("#myGraph")
  .selectAll("rect")   // rect 요소 지정
  .data(dataSet)    // 데이터 설정
// 히트맵 표시
heatMap.enter()
  .append("rect")  // rect 요소를 추가
  .attr("class", "block")	// CSS 클래스 추가
  .attr("x", function(d, i) { // X 좌표 설정
		return (i % 8) * blockSize;
	}) 
  .attr("y", function(d, i) { // Y 좌표 설정
		return Math.floor(i/8)*blockSize;
	})
  .attr("width", function(d, i) {	// 넓이 설정
		return blockSize;
	})
  .attr("height", function(d, i) {	// 높이 설정
		return blockSize;
	})
  .style("fill", function(d, i){	// 색 표시
		return color(d/maxValue);
	})
// 히트맵을 일정 간견으로 갱신
setInterval(function(){
	for(var i=0; i<dataSet.length; i++){
		var n = ((Math.random() * 3.5) | 0) - 2;	// 난수 값
		dataSet[i] = dataSet[i] + n;	// 더하기
		if (dataSet[i] < 0){ dataSet[i] = 0; }	// 음수가 되지 않도록 조정
		if (dataSet[i] > maxValue ){ dataSet[i] = maxValue; }	// 최댓값을 넘지 않도록 조정
	}
	heatMap.data(dataSet)
	  .style("fill", function(d, i){	// 색 표시
			return color(d/maxValue);
		})
}, 2000);
