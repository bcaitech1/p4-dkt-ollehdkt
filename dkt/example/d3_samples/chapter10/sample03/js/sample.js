var svgWidth = 320;	// SVG 요소의 넓이
var svgHeight = 240;	// SVG 요소의 높이
var offsetX = 30;	// X 좌표의 오프셋
var offsetY = 20;	// Y 좌표의 오프셋
// 데이터셋
var dataSet = [
		[30, 40], [120, 115], [125, 90], [150, 160], [300, 170],
		[60, 40], [140, 145], [165, 110], [200, 170], [250, 160]
	];
// 산포도 그리기
var circleElements = d3.select("#myGraph")
  .selectAll("circle")	// circle 요소 추가
  .data(dataSet)	// 데이터셋을 요소에 설정
circleElements
  .enter()
  .append("circle")	// 데이터의 개수만큼 circle 요소가 추가됨
  .attr("class", "mark")	// CSS 클래스 지정
  .attr("cx", function(d, i){
		return d[0] + offsetX;	// 최초 요소를 X 좌표로 함
	})
  .attr("cy", function(d, i){
		return svgHeight-d[1] - offsetY;	// 2번째의 요소를 Y 좌표로 함
	})
  .attr("r", 5)	// 반지름을 지정
// 데이터셋 갱신
function updateData(data){
	var result = data.map(function(d, i){	// 배열 요소 수만큼 반복
		var x = Math.random() * svgWidth;	// 난수 생성
		var y = Math.random() * svgHeight;
		return [x, y];	// 좌표값을 배열로 반환
	})
	return result;
}
// 산포도 갱신
function updateGraph(){
	circleElements
	  .data(dataSet)
	  .transition()	// cx, cy 애니메이션
	  .attr("cx", function(d, i){
			return d[0] + offsetX;	// X 좌표를 설정
		})
	  .attr("cy", function(d, i){
			return svgHeight-d[1] - offsetY;	// Y 좌표를 설정
		})
}
// 눈금 표시
function drawScale(){
	d3.select("#myGraph").selectAll("g").remove(); //눈금 요소 삭제 ● 수정
	var maxX = d3.max(dataSet, function(d, i){
		return d[0];	// X 좌표값
	});
	var maxY = d3.max(dataSet, function(d, i){
		return d[1];	// Y 좌표값
	});
	// 세로 눈금을 표시하고자 D3 스케일을 설정
	var yScale = d3.scale.linear()  // 스케일 설정
	  .domain([0, maxY])   // 원래 데이터 범위
	  .range([maxY, 0]) // 실제 표시 크기
	// 눈금 표시
	d3.select("#myGraph")	// SVG 요소를 지정
	  .append("g")	// g 요소 추가. 이것이 눈금을 표시하는 요소가 됨
	  .attr("class", "axis")	// CSS 클래스 지정
	  .attr("transform", "translate("+offsetX+", "+(svgHeight-maxY-offsetY)+")")
	  .call(
			d3.svg.axis()
		  .scale(yScale)  //스케일 적용
		  .orient("left") //눈금 표시 위치를 왼쪽으로 지정
		)
	// 가로 눈금을 표시하고자 D3 스케일 설정
	var xScale = d3.scale.linear()  // 스케일 설정
	  .domain([0, maxX])   // 원래 데이터 범위
	  .range([0, maxX]) // 실제 표시 크기
	// 눈금 표시
	d3.select("#myGraph")	// SVG 요소를 지정
		  .append("g")	// g 요소 추가. 이것이 눈금을 표시하는 요소가 됨
		  .attr("class", "axis")	// CSS 클래스 지정
		  .attr("transform", "translate("+offsetX+", "+(svgHeight-offsetY)+")")
		  .call(
				d3.svg.axis()
			  .scale(xScale)  //스케일 적용
			  .orient("bottom") //눈금 표시 위치를 왼쪽으로 지정
			)
}
// 눈금 표시
drawScale();
// 타이머를 사용하여 2초마다 단위를 변화시킴
setInterval(function(){
	dataSet = updateData(dataSet);	// 데이터 갱신
	drawScale(); // 눈금 그리기 ● 수정
	updateGraph();	// 그래프 갱신
}, 2000);
