var svgWidth = 320;	// SVG 요소의 넓이
var svgHeight = 300;	// SVG 요소의 높이
var offsetX = 30;	// X 좌표의 오프셋
var offsetY = 20;	// Y 좌표의 오프셋
var svg = d3.select("#myGraph");	// SVG 요소를 지정
// 데이터셋
var dataSet = [
		[30, 40], [120, 115], [125, 90], [150, 160], [300, 170],
		[60, 40], [140, 145], [165, 110], [200, 260], [280, 160]
	];
// 산포도 그리기
var circleElements = svg
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
	var maxX = d3.max(dataSet, function(d, i){
		return d[0];	// X 좌표값
	});
	var maxY = d3.max(dataSet, function(d, i){
		return d[1];	// Y 좌표값
	});
	// 세로 눈금을 표시하고자 D3 스케일을 설정
	var yScale = d3.scale.linear()  // 스케일 설정
	  .domain([0, maxY])   // 원래 크기
	  .range([maxY, 0]) // 실제 표시 크기
	// 눈금 표시
	svg.append("g")	// g 요소 추가. 이것이 눈금을 표시하는 요소가 됨
	  .attr("class", "axis")	// CSS 클래스 지정
	  .attr("transform", "translate("+offsetX+", "+(svgHeight-maxY-offsetY)+")")
	  .call(
			d3.svg.axis()
		  .scale(yScale)  //스케일 적용
		  .orient("left") //눈금 표시 위치를 왼쪽으로 지정
		)
	// 가로 눈금을 표시하고자 D3 스케일 설정
	var xScale = d3.scale.linear()  // 스케일 설정
	  .domain([0, maxX])   // 원래 크기
	  .range([0, maxX]) // 실제 표시 크기
	// 눈금 표시
	svg.append("g")	// g 요소 추가. 이것이 눈금을 표시하는 요소가 됨
		  .attr("class", "axis")	// CSS 클래스 지정
		  .attr("transform", "translate("+offsetX+", "+(svgHeight-offsetY)+")")
		  .call(
				d3.svg.axis()
			  .scale(xScale)  //스케일 적용
			  .orient("bottom") //눈금 표시 위치를 왼쪽으로 지정
			)
	// 그리드 표시
	var grid = svg.append("g");
	// 가로 방향과 세로 방향의 그리드 간격 자동 생성
	var rangeX = d3.range(50, maxX, 50);
	var rangeY = d3.range(20, maxY, 20);
	// 세로 방향 그리드 생성
	grid.selectAll("line.y")	// line요소의 y 클래스를 선택
	  .data(rangeY)	// 세로 방향의 범위를 데이터셋으로 설정
	  .enter()
	  .append("line")	// line 요소 추가
	  .attr("class", "grid")	// CSS 클래스의 grid를 지정
		// (x1,y1)-(x2,y2)의 좌표값을 설정
	  .attr("x1", offsetX)
	  .attr("y1", function(d, i){
			return svgHeight - d - offsetY;
		})
	  .attr("x2", maxX + offsetX)
	  .attr("y2", function(d, i){
			return svgHeight - d - offsetY;
		})
	// 가로 방향의 그리드 생성
	grid.selectAll("line.x")	// line요소의 x 클래스를 선택
	  .data(rangeX)	// 가로 방향의 범위를 데이터셋으로 설정
	  .enter()
	  .append("line")	// line 요소 추가
	  .attr("class", "grid")	// CSS 클래스의 grid를 지정
		// (x1,y1)-(x2,y2)의 좌표값을 설정
	  .attr("x1", function(d, i){
			return d + offsetX;
		})
	  .attr("y1", svgHeight - offsetY)
	  .attr("x2", function(d, i){
			return d + offsetX;
		})
	  .attr("y2", svgHeight -offsetY - maxY)
}
// 풍선 도움말을 생성
var tooltip = d3.select("body")
  .append("div")
  .attr("class", "tip")
// 풍선 도움말을 표시
circleElements
  .on("mouseover", function(d){
		var x = parseInt(d[0]);	// 원의 X 좌표값
		var y = parseInt(d[1]);	// 원의 Y 좌표값
		var data = d3.select(this).datum();	// 요소의 데이터를 읽어옴
		var dx = parseInt(data[0]);	// 원의 X 좌표. 변수 x와 같은 값
		var dy = parseInt(data[1]);	// 원의 Y 좌표. 변수 y와 같은 값
		tooltip
		  .style("left", offsetX + x+"px")
		  .style("top", svgHeight + offsetY - y+"px")
		  .style("visibility", "visible")	// 풍선 도움말을 표시
		  .text(dx+", "+dy)
	})
  .on("mouseout", function(){
		tooltip.style("visibility", "hidden")	// 풍선 도움말을 숨김
	})
// 눈금과 그리드를 표시
drawScale();
// 타이머를 사용하여 4초마다 위치를 바꿈　●변경
setInterval(function(){
	dataSet = updateData(dataSet);	// 데이터 갱신
	updateGraph();	// 그래프 갱신
}, 4000);

