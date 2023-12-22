import { LX } from 'lexgui';

const Rooms = window.Rooms = {

    dragSupportedExtensions: [ 'hdre', 'glb' ],

    init() {

        this.initUI();
    },

    initUI() {

        var area = LX.init();

        var canvas = document.getElementById( "canvas" );
        area.attach( canvas );

        canvas.addEventListener('dragenter', e => e.preventDefault() );
        canvas.addEventListener('dragleave', e => e.preventDefault() );
        canvas.addEventListener('drop', (e) => {
            e.preventDefault();
            const file = e.dataTransfer.files[0];
            const ext = LX.getExtension( file.name );
            if( this.dragSupportedExtensions.indexOf( ext ) == -1 )
                return;
            switch( ext ) {
                case 'hdre': this.loadEnvironment( file ); break;
                case 'glb': this.loadGltf( file ); break;
            }
        });

        new LX.PocketDialog( "Control Panel", p => {

            p.branch( "Digital Location", { closed: true } );
            p.addText( "Name", "", null, { signal: "@location_name", disabled: true } );
            p.addFile( "Load", (data, file) => this.loadGltf(data, file), { type: 'buffer', local: false } );
            p.addCheckbox( "Rotate", false, () => Module.toggleSceneRotation() );
        
            p.branch( "Environment", { closed: true } );
            p.addText( "Name", "", null, { signal: "@environment_name", disabled: true } );
            p.addFile( "Load", (data, file) => this.loadEnvironment(data, file), { type: 'buffer', local: false } );
        
            p.branch( "Camera", { closed: true } );
            p.addDropdown( "Type", [ "Flyover", "Orbit" ], "Orbit" );
        
        }, { size: [300, null], float: "left", draggable: false });

    },

    loadEnvironment( data, file ) {

        if( data.constructor === File )
        {
            const reader = new FileReader();
            reader.readAsArrayBuffer( data );
            reader.onload = e => this._loadEnvironment( data.name, e.target.result );
            return;
        }
        
        this._loadEnvironment( file.name, data );
    },

    loadGltf( data, file ) {

        if( data.constructor === File )
        {
            const reader = new FileReader();
            reader.readAsArrayBuffer( data );
            reader.onload = e => this._loadGltf( data.name, e.target.result );
            return;
        }
        
        this._loadGltf( file.name, data );
    },

    _loadEnvironment( name, buffer ) {

        name = name.substring( name.lastIndexOf( '/' ) );
        
        console.log( "Loading hdre", [ name, buffer ] );

        this._fileStore( name, buffer );

        // This will load the hdre and set texture to the skybox
        Module.Engine.setEnvironment( name );

        // Update UI
        LX.emit( '@environment_name', name.replace( '.hdre', '' ) );
    },

    _loadGltf( name, buffer ) {

        name = name.substring( name.lastIndexOf( '/' ) );
        
        console.log( "Loading glb", [ name, buffer ] );

        this._fileStore( name, buffer );

        Module.Engine.loadGLB( name );

        // Update UI
        LX.emit( '@location_name', name.replace( '.glb', '' ) );
    },

    _fileStore( filename, buffer ) {

        let data = new Uint8Array( buffer );
        let stream = FS.open( filename, 'w+' );
        FS.write( stream, data, 0, data.length, 0 );
        FS.close( stream );
    }

};

Rooms.init();