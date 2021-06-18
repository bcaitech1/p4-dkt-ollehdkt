// WebGL 관련 처리
var webGLWidth = 320;	// Canvas(WebGL) 요소의 넓이
var webGLHeight = 320;	// Canvas(WebGL) 요소의 높이
// 그리기 영역, 카메라, 조명 설정
renderer = new THREE.WebGLRenderer({ antialias: true});
renderer.setSize( webGLWidth, webGLHeight );	// 크기 지정
renderer.setClearColor(0xffffaf);	// 배경색을 지정
document.body.appendChild( renderer.domElement );	// 페이지 끝에 추가
var camera = new THREE.PerspectiveCamera( 70, 1.0, 1, 1000 );	// 카메라 각도 등 설정
camera.position.set(0, 0, 70);	// 카메라 위치 설정
var scene = new THREE.Scene();	// 장면 생성
var cube = [ ];	// 육면체를 넣을 배열
var light = new THREE.DirectionalLight(0xffffff, 1.25);	// 조명 생성
light.position.set(70, 120, 2000);	// 조명 위치 설정
scene.add(light);	// 조명 추가
// 이하는 D3.js 처리
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
	.selectAll("div")   // div에 표시할 상자 할당
	.data(dataSet)    // 데이터 설정
// 히트맵 표시
heatMap.enter()
	.call(function(d){	// 데이터의 수만큼 육면체 생성
		var count = 0;
		for(var y=0; y<12; y++){	// 세로 생성 수
			for(var x =0; x<8; x++){	// 가로 생성 수
				cube[count] = new THREE.Mesh(	// 육면체의 머티리얼 등 지정
					new THREE.BoxGeometry( 5, 5, 5 ),	// 육면체의 크기
					new THREE.MeshLambertMaterial( { color : 0x00ff00 } )	// Lambert로 색을 지정
				);
				cube[count].position.x = x * 8 - 25;	// X 좌표 설정
				cube[count].position.y = y * 8 - 45;	// y 좌표 설정
				scene.add(cube[count]);	// 장면 추가
				count = count + 1;
			}
		}
	})
// 히트맵을 일정 간격으로 갱신
setInterval(function(){
	for(var i=0; i<dataSet.length; i++){
		var n = ((Math.random() * 3.5) | 0) - 2;	// 난수 값
		dataSet[i] = dataSet[i] + n;	// 더하기
		if (dataSet[i] < 0){ dataSet[i] = 0; }	// 음수가 되지 않도록 조정
		if (dataSet[i] > maxValue ){ dataSet[i] = maxValue; }	// 최댓값을 넘지 않도록 조정
	}
	heatMap.data(dataSet);	// 데이터 설정
	// 데이터의 수만큼 육면체의 색을 변경
	for(var i=0; i<dataSet.length; i++){
		cube[i].material.color = new THREE.Color(color(dataSet[i]/maxValue));	// 머티리얼 색을 설정
	}
}, 1000);
// 육면체의 회전은 별도 애니메이션으로 처리
cubeAnime();
function cubeAnime(){
	requestAnimationFrame(cubeAnime);	// 그리기 타이밍에 맞추어 함수를 호출
	for(var i=0; i<cube.length; i++){
		cube[i].rotation.x = cube[i].rotation.x - 0.01;	// X 축의 회전 각도
		cube[i].rotation.y = cube[i].rotation.y + 0.01;	// Y 축의 회전 각도
		cube[i].rotation.z = cube[i].rotation.z + 0.02;	// Z 축의 회전 각도
	}
	camera.rotation.z = camera.rotation.z + 0.01;	// 카메라 Z 축의 회전 각도
	renderer.render( scene, camera );	// 장면 그리기
}
