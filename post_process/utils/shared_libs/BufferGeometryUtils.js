( function () {

	function mergeBufferGeometries( geometries ) {

		var attributeNames = new Set();

		for ( var i = 0; i < geometries.length; i ++ ) {

			var geometry = geometries[ i ];

			for ( var name in geometry.attributes ) {

				attributeNames.add( name );

			}

		}

		var mergedGeometry = new THREE.BufferGeometry();

		attributeNames.forEach( function ( name ) {

			var arrays = [];
			var itemSize = 0;
			var normalized = false;

			for ( var i = 0; i < geometries.length; i ++ ) {

				var attribute = geometries[ i ].attributes[ name ];

				if ( attribute === undefined ) {

					console.error( 'THREE.BufferGeometryUtils.mergeBufferGeometries: All geometries must have matching attributes; "' + name + '" missing in geometry ' + i );
					return null;

				}

				arrays.push( attribute.array );
				itemSize = attribute.itemSize;
				normalized = attribute.normalized;

			}

			var totalLength = 0;

			for ( var i = 0; i < arrays.length; i ++ ) {

				totalLength += arrays[ i ].length;

			}

			var mergedArray = new Float32Array( totalLength );
			var offset = 0;

			for ( var i = 0; i < arrays.length; i ++ ) {

				mergedArray.set( arrays[ i ], offset );
				offset += arrays[ i ].length;

			}

			mergedGeometry.setAttribute( name, new THREE.BufferAttribute( mergedArray, itemSize, normalized ) );

		} );

		return mergedGeometry;

	}

	THREE.BufferGeometryUtils = {
		mergeBufferGeometries: mergeBufferGeometries
	};

} )();
