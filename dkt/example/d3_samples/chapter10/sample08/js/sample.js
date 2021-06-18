// CSV 데이터를 불러옴
d3.csv("mydata.csv", function(error, data){
	// SVG 요소의 넓이와 높이를 구함
	var svgEle = document.getElementById("myGraph");
	var svgWidth = window.getComputedStyle(svgEle, null).getPropertyValue("width");
	var svgHeight = window.getComputedStyle(svgEle, null).getPropertyValue("height");
	svgWidth = parseFloat(svgWidth);	// 값에는 단위가 있으므로 단위를 삭제
	svgHeight = parseFloat(svgHeight);	// 값에는 단위가 있으므로 단위를 삭제
	var offsetX = 30;	// X 좌표의 오프셋　●
	var offsetY = 20;	// Y 좌표의 오프셋　●
	var xScale;	// 가로축의 스케일●
	var yScale;	// 세로축의 스케일　●
	var yAxisHeight = svgHeight - 20;	// 가로축의 넓이　●
	var xAxisWidth = svgWidth - 40;	// 세로축의 넓이　●
	var svg = d3.select("#myGraph");	// SVG 요소를 지정
	// 데이터를 불러와 설정함
	var dataSet = [ ];	// 데이터를 저장할 배열 변수
	data.forEach(function(d, i){	// 배열 데이터 읽어오기
		dataSet.push([d.total/100, d.bug*1, d.time*1]);	// *1로 숫자 변환
	});
	// 눈금과 그리드를 먼저 표시　●
	drawScale();
	// 산포도 그리기
	var circleElements = svg
	  .selectAll("circle")	// circle 요소 추가
	  .data(dataSet)	// 데이터셋을 요소에 설정
	circleElements
	  .enter()
	  .append("circle")	// 데이터의 개수만큼 circle 요소가 추가됨
	  .attr("class", "mark")	// CSS 클래스 지정
	  .attr("cx", svgWidth / 2 + offsetX)	// X 좌표의 눈금 가운데로 함
	  .attr("cy", svgHeight/2 - offsetY)	// Y 좌표의 눈금 가운데로 함
	  .attr("r", 100)	// 반지름을 100으로 함
	  .attr("opacity", 0)	// 투명하게 함
	  .transition()	// X,Y 좌표까지 애니메이션
	  .duration(2000)	// 2초 동안 애니메이션
	  .ease("bounce")	// 통통튀는 움직임으로 함
	  .attr("cx", function(d, i){
			return xScale(d[0]) + offsetX;	// 최초 요소를 X 좌표로 함　●
		})
	  .attr("cy", function(d, i){
			return yScale(d[1]);	// 2번째의 요소를 Y 좌표로 함　●
		})
	  .attr("r", 5)	// 반지름을 지정
	  .attr("opacity", 1.0)	// 불투명하게 함
	// 눈금 표시
	function drawScale(){
		var maxX = d3.max(dataSet, function(d, i){
			return d[0];	// X 좌표값
		});
		var maxY = d3.max(dataSet, function(d, i){
			return d[1];	// Y 좌표값
		});
		// 세로 눈금을 표시하고자 D3 스케일을 설정
		yScale = d3.scale.linear()  // 스케일 설정　●
		  .domain([0, maxY])   // 원래 크기
		  .range([yAxisHeight, 0]) // 실제 표시 크기　●
		// 눈금 표시
		svg.append("g")	// g 요소 추가. 이것이 눈금을 표시하는 요소가 됨
			  .attr("class", "axis")	// CSS 클래스 지정
			  .attr("transform", "translate("+offsetX+", "+(svgHeight-yAxisHeight-offsetY)+")")	// ●
			  .call(
					d3.svg.axis()
				  .scale(yScale)  //스케일 적용
				  .orient("left") //눈금 표시 위치를 왼쪽으로 지정
				)
		// 가로 눈금을 표시하고자 D3 스케일 설정
		xScale = d3.scale.linear()  // 스케일 설정　●
		  .domain([0, maxX])   // 원래 크기
		  .range([0, xAxisWidth]) // 실제 표시 크기　●
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
		var rangeX = d3.range(50, maxX+50, 50);
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
				return svgHeight - yScale(d) - offsetY;	// ●
			})
		  .attr("x2", xAxisWidth + offsetX)
		  .attr("y2", function(d, i){
				return svgHeight - yScale(d) - offsetY;	// ●
			})
		// 가로 방향의 그리드 생성
		grid.selectAll("line.x")	// line요소의 x 클래스를 선택
		  .data(rangeX)	// 가로 방향의 범위를 데이터셋으로 설정
		  .enter()
		  .append("line")	// line 요소 추가
		  .attr("class", "grid")	// CSS 클래스의 grid를 지정
			// (x1,y1)-(x2,y2)의 좌표값을 설정
		  .attr("x1", function(d, i){
				return xScale(d) + offsetX;	//  ●
			})
		  .attr("y1", svgHeight - offsetY)
		  .attr("x2", function(d, i){
				return xScale(d) + offsetX;	//  ●
			})
		  .attr("y2", svgHeight -offsetY - yAxisHeight)	// ●
	}
	// 풍선 도움말을 생성
	var tooltip = d3.select("body")
	  .append("div")
	  .attr("class", "tip")
	// 풍선 도움말에 시간을 표시
	circleElements
	  .on("mouseover", function(d){
			var x = parseInt(xScale(d[0]));	// 원의 X 좌표값  ●
			var y = parseInt(yScale(d[1]));	// 원의 Y 좌표값  ●
			var t = parseInt(d[2]);	// 해결에 필요한 시간  ●
			tooltip
			  .style("left", offsetX + x+"px")
			  .style("top", offsetY + 30 + y+"px")	//　●
			  .style("visibility", "visible")	// 풍선 도움말을 표시
			  .text(t+"시간")
		})
	  .on("mouseout", function(){
			tooltip.style("visibility", "hidden")	// 풍선 도움말을 숨김
		})
});
