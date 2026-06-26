( function () {

	/**
	 * Minimal GLTFLoader for Three.js r128.
	 *
	 * Supports loading .glb (binary glTF) files with basic mesh geometry.
	 * Handles positions, normals, indices, UV coordinates, and vertex colors.
	 * Materials are created as MeshStandardMaterial with PBR metallic-roughness defaults.
	 *
	 * Usage:
	 *   const loader = new THREE.GLTFLoader();
	 *   loader.load( 'model.glb', function ( gltf ) {
	 *       scene.add( gltf.scene );
	 *   } );
	 */

	// glTF constants
	var WEBGL_COMPONENT_TYPES = {
		5120: Int8Array,
		5121: Uint8Array,
		5122: Int16Array,
		5123: Uint16Array,
		5125: Uint32Array,
		5126: Float32Array
	};

	var WEBGL_COMPONENT_TYPE_SIZES = {
		5120: 1,
		5121: 1,
		5122: 2,
		5123: 2,
		5125: 4,
		5126: 4
	};

	var ACCESSOR_TYPE_SIZES = {
		'SCALAR': 1,
		'VEC2': 2,
		'VEC3': 3,
		'VEC4': 4,
		'MAT2': 4,
		'MAT3': 9,
		'MAT4': 16
	};

	var GLB_HEADER_MAGIC = 0x46546C67; // 'glTF'
	var GLB_HEADER_LENGTH = 12;
	var GLB_CHUNK_TYPE_JSON = 0x4E4F534A;
	var GLB_CHUNK_TYPE_BIN = 0x004E4942;

	// ---------- GLB binary parser ----------

	function parseGLB( data ) {

		var headerView = new DataView( data, 0, GLB_HEADER_LENGTH );

		var magic = headerView.getUint32( 0, true );

		if ( magic !== GLB_HEADER_MAGIC ) {

			throw new Error( 'GLTFLoader: Unsupported glTF-Binary header.' );

		}

		var version = headerView.getUint32( 4, true );

		if ( version < 2 ) {

			throw new Error( 'GLTFLoader: Legacy binary file detected.' );

		}

		var jsonChunk = null;
		var binChunk = null;

		var chunkIndex = GLB_HEADER_LENGTH;

		while ( chunkIndex < data.byteLength ) {

			var chunkView = new DataView( data, chunkIndex, 8 );
			var chunkLength = chunkView.getUint32( 0, true );
			var chunkType = chunkView.getUint32( 4, true );

			if ( chunkType === GLB_CHUNK_TYPE_JSON ) {

				var jsonArray = new Uint8Array( data, chunkIndex + 8, chunkLength );
				var jsonText = '';

				for ( var i = 0; i < jsonArray.length; i ++ ) {

					jsonText += String.fromCharCode( jsonArray[ i ] );

				}

				jsonChunk = JSON.parse( jsonText );

			} else if ( chunkType === GLB_CHUNK_TYPE_BIN ) {

				binChunk = data.slice( chunkIndex + 8, chunkIndex + 8 + chunkLength );

			}

			chunkIndex += 8 + chunkLength;

		}

		return { json: jsonChunk, bin: binChunk };

	}

	// ---------- Accessor data extraction ----------

	function getAccessorData( json, accessorIndex, binData ) {

		var accessor = json.accessors[ accessorIndex ];
		var bufferView = json.bufferViews[ accessor.bufferView ];

		var itemSize = ACCESSOR_TYPE_SIZES[ accessor.type ];
		var TypedArrayClass = WEBGL_COMPONENT_TYPES[ accessor.componentType ];
		var componentSize = WEBGL_COMPONENT_TYPE_SIZES[ accessor.componentType ];

		var byteOffset = ( bufferView.byteOffset || 0 ) + ( accessor.byteOffset || 0 );
		var count = accessor.count;
		var byteStride = bufferView.byteStride;

		var array;

		if ( byteStride && byteStride !== itemSize * componentSize ) {

			// Interleaved data -- must de-interleave
			array = new TypedArrayClass( count * itemSize );
			var src = new DataView( binData );
			var elementsPerStride = byteStride / componentSize;

			for ( var i = 0; i < count; i ++ ) {

				for ( var j = 0; j < itemSize; j ++ ) {

					var srcIndex = byteOffset + i * byteStride + j * componentSize;

					if ( componentSize === 4 ) {

						if ( accessor.componentType === 5126 ) {

							array[ i * itemSize + j ] = src.getFloat32( srcIndex, true );

						} else {

							array[ i * itemSize + j ] = src.getUint32( srcIndex, true );

						}

					} else if ( componentSize === 2 ) {

						array[ i * itemSize + j ] = src.getUint16( srcIndex, true );

					} else {

						array[ i * itemSize + j ] = new Uint8Array( binData )[ srcIndex ];

					}

				}

			}

		} else {

			// Contiguous data
			array = new TypedArrayClass( binData, byteOffset, count * itemSize );

		}

		return { array: array, itemSize: itemSize, count: count, componentType: accessor.componentType, type: accessor.type };

	}

	// ---------- Material creation ----------

	function createMaterial( json, materialIndex ) {

		if ( materialIndex === undefined || ! json.materials || ! json.materials[ materialIndex ] ) {

			return new THREE.MeshStandardMaterial( { color: 0xcccccc, metalness: 0.0, roughness: 0.7 } );

		}

		var matDef = json.materials[ materialIndex ];
		var params = {};

		params.name = matDef.name || '';

		if ( matDef.pbrMetallicRoughness ) {

			var pbr = matDef.pbrMetallicRoughness;

			if ( pbr.baseColorFactor ) {

				var c = pbr.baseColorFactor;
				params.color = new THREE.Color( c[ 0 ], c[ 1 ], c[ 2 ] );

				if ( c[ 3 ] < 1.0 ) {

					params.transparent = true;
					params.opacity = c[ 3 ];

				}

			}

			params.metalness = pbr.metallicFactor !== undefined ? pbr.metallicFactor : 1.0;
			params.roughness = pbr.roughnessFactor !== undefined ? pbr.roughnessFactor : 1.0;

		} else {

			params.metalness = 0.0;
			params.roughness = 0.7;

		}

		if ( matDef.doubleSided === true ) {

			params.side = THREE.DoubleSide;

		}

		if ( matDef.alphaMode === 'BLEND' ) {

			params.transparent = true;

		}

		if ( matDef.emissiveFactor ) {

			var e = matDef.emissiveFactor;
			params.emissive = new THREE.Color( e[ 0 ], e[ 1 ], e[ 2 ] );

		}

		return new THREE.MeshStandardMaterial( params );

	}

	// ---------- Mesh building ----------

	function buildMesh( json, meshDef, binData ) {

		var group = new THREE.Group();
		group.name = meshDef.name || '';

		var primitives = meshDef.primitives;

		for ( var p = 0; p < primitives.length; p ++ ) {

			var primitive = primitives[ p ];
			var geometry = new THREE.BufferGeometry();

			// Attributes
			var attributes = primitive.attributes;

			if ( attributes.POSITION !== undefined ) {

				var posData = getAccessorData( json, attributes.POSITION, binData );
				geometry.setAttribute( 'position', new THREE.BufferAttribute( new Float32Array( posData.array ), posData.itemSize ) );

				// Use accessor min/max for bounding box if available
				var posAccessor = json.accessors[ attributes.POSITION ];

				if ( posAccessor.min && posAccessor.max ) {

					geometry.boundingBox = new THREE.Box3(
						new THREE.Vector3().fromArray( posAccessor.min ),
						new THREE.Vector3().fromArray( posAccessor.max )
					);

				}

			}

			if ( attributes.NORMAL !== undefined ) {

				var normData = getAccessorData( json, attributes.NORMAL, binData );
				geometry.setAttribute( 'normal', new THREE.BufferAttribute( new Float32Array( normData.array ), normData.itemSize ) );

			}

			if ( attributes.TEXCOORD_0 !== undefined ) {

				var uvData = getAccessorData( json, attributes.TEXCOORD_0, binData );
				geometry.setAttribute( 'uv', new THREE.BufferAttribute( new Float32Array( uvData.array ), uvData.itemSize ) );

			}

			if ( attributes.COLOR_0 !== undefined ) {

				var colData = getAccessorData( json, attributes.COLOR_0, binData );
				var colArray;

				// Normalize if integer type
				if ( colData.componentType === 5121 ) {

					colArray = new Float32Array( colData.array.length );

					for ( var ci = 0; ci < colData.array.length; ci ++ ) {

						colArray[ ci ] = colData.array[ ci ] / 255.0;

					}

				} else if ( colData.componentType === 5123 ) {

					colArray = new Float32Array( colData.array.length );

					for ( var ci = 0; ci < colData.array.length; ci ++ ) {

						colArray[ ci ] = colData.array[ ci ] / 65535.0;

					}

				} else {

					colArray = new Float32Array( colData.array );

				}

				geometry.setAttribute( 'color', new THREE.BufferAttribute( colArray, colData.itemSize ) );

			}

			// Indices
			if ( primitive.indices !== undefined ) {

				var idxData = getAccessorData( json, primitive.indices, binData );
				var indexArray;

				if ( idxData.componentType === 5125 ) {

					indexArray = new Uint32Array( idxData.array );

				} else if ( idxData.componentType === 5123 ) {

					indexArray = new Uint16Array( idxData.array );

				} else {

					indexArray = new Uint8Array( idxData.array );

				}

				geometry.setIndex( new THREE.BufferAttribute( indexArray, 1 ) );

			}

			// Compute normals if not provided
			if ( attributes.NORMAL === undefined ) {

				geometry.computeVertexNormals();

			}

			// Material
			var material = createMaterial( json, primitive.material );

			// Enable vertex colors if present
			if ( attributes.COLOR_0 !== undefined ) {

				material.vertexColors = true;

			}

			// Create mesh
			var mode = primitive.mode !== undefined ? primitive.mode : 4; // default TRIANGLES

			var mesh;

			if ( mode === 0 ) {

				mesh = new THREE.Points( geometry, material );

			} else if ( mode === 1 || mode === 2 || mode === 3 ) {

				mesh = new THREE.LineSegments( geometry, material );

			} else {

				mesh = new THREE.Mesh( geometry, material );

			}

			mesh.name = meshDef.name || '';

			group.add( mesh );

		}

		// If only one primitive, return the child directly (set on the group level later)
		if ( group.children.length === 1 ) {

			var child = group.children[ 0 ];
			child.name = group.name;
			return child;

		}

		return group;

	}

	// ---------- Node tree building ----------

	function buildNodeHierarchy( json, binData ) {

		var nodes = [];

		if ( ! json.nodes ) return nodes;

		// First pass: create objects for each node
		for ( var i = 0; i < json.nodes.length; i ++ ) {

			var nodeDef = json.nodes[ i ];
			var node;

			if ( nodeDef.mesh !== undefined ) {

				node = buildMesh( json, json.meshes[ nodeDef.mesh ], binData );

			} else {

				node = new THREE.Group();

			}

			node.name = nodeDef.name || ( 'node_' + i );

			// Apply TRS
			if ( nodeDef.matrix ) {

				var m = new THREE.Matrix4();
				m.fromArray( nodeDef.matrix );
				m.decompose( node.position, node.quaternion, node.scale );

			} else {

				if ( nodeDef.translation ) {

					node.position.fromArray( nodeDef.translation );

				}

				if ( nodeDef.rotation ) {

					node.quaternion.fromArray( nodeDef.rotation );

				}

				if ( nodeDef.scale ) {

					node.scale.fromArray( nodeDef.scale );

				}

			}

			nodes.push( node );

		}

		// Second pass: build parent-child relationships
		for ( var i = 0; i < json.nodes.length; i ++ ) {

			var nodeDef = json.nodes[ i ];

			if ( nodeDef.children ) {

				for ( var j = 0; j < nodeDef.children.length; j ++ ) {

					nodes[ i ].add( nodes[ nodeDef.children[ j ] ] );

				}

			}

		}

		return nodes;

	}

	function buildScene( json, binData, sceneIndex ) {

		var sceneDef = json.scenes[ sceneIndex ];
		var scene = new THREE.Group();
		scene.name = sceneDef.name || 'Scene';

		var nodes = buildNodeHierarchy( json, binData );

		if ( sceneDef.nodes ) {

			for ( var i = 0; i < sceneDef.nodes.length; i ++ ) {

				scene.add( nodes[ sceneDef.nodes[ i ] ] );

			}

		}

		return { scene: scene, nodes: nodes };

	}

	// ---------- GLTFLoader class ----------

	class GLTFLoader extends THREE.Loader {

		constructor( manager ) {

			super( manager );

		}

		load( url, onLoad, onProgress, onError ) {

			var scope = this;

			var loader = new THREE.FileLoader( this.manager );
			loader.setPath( this.path );
			loader.setResponseType( 'arraybuffer' );
			loader.setRequestHeader( this.requestHeader );
			loader.setWithCredentials( this.withCredentials );

			loader.load( url, function ( data ) {

				try {

					scope.parse( data, '', function ( result ) {

						onLoad( result );

					}, onError );

				} catch ( e ) {

					if ( onError ) {

						onError( e );

					} else {

						console.error( e );

					}

					scope.manager.itemError( url );

				}

			}, onProgress, onError );

		}

		parse( data, path, onLoad, onError ) {

			try {

				var glb = parseGLB( data );
				var json = glb.json;
				var binData = glb.bin;

				if ( ! json.asset || json.asset.version[ 0 ] < '2' ) {

					if ( onError ) {

						onError( new Error( 'GLTFLoader: Unsupported asset version: ' + ( json.asset ? json.asset.version : 'unknown' ) ) );

					}

					return;

				}

				var sceneIndex = json.scene !== undefined ? json.scene : 0;

				// Build all scenes
				var scenes = [];
				var mainScene = null;

				if ( json.scenes && json.scenes.length > 0 ) {

					for ( var i = 0; i < json.scenes.length; i ++ ) {

						var built = buildScene( json, binData, i );
						scenes.push( built.scene );

						if ( i === sceneIndex ) {

							mainScene = built.scene;

						}

					}

				} else {

					// No scenes defined -- build meshes directly
					mainScene = new THREE.Group();
					mainScene.name = 'Scene';

					if ( json.meshes ) {

						for ( var i = 0; i < json.meshes.length; i ++ ) {

							mainScene.add( buildMesh( json, json.meshes[ i ], binData ) );

						}

					}

					scenes.push( mainScene );

				}

				// Collect cameras (minimal -- just note their existence)
				var cameras = [];

				var result = {
					scene: mainScene,
					scenes: scenes,
					cameras: cameras,
					animations: [],
					asset: json.asset || {}
				};

				onLoad( result );

			} catch ( e ) {

				if ( onError ) {

					onError( e );

				} else {

					throw e;

				}

			}

		}

	}

	THREE.GLTFLoader = GLTFLoader;

} )();
