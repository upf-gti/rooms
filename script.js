import { LX } from 'lexgui';

const Rooms = window.Rooms = {

    dragSupportedExtensions: [ 'hdre', 'glb' ],

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
            p.addText( "Name", "placeholder", null, { signal: "@location_name", disabled: true } );
            p.addFile( "Load", (e) => { console.log(e) }, { type: 'bin', local: false } );
            p.addCheckbox( "Rotate", false, null, { signal: "@location_name" } );
        
            p.branch( "Environment", { closed: true } );
            p.addText( "Name", "placeholder", null, { signal: "@environment_name", disabled: true } );
            p.addFile( "Load", (e) => this.loadEnvironment(e), { type: 'bin', local: false } );
        
            p.branch( "Camera", { closed: true } );
            p.addDropdown( "Type", [ "Flyover", "Orbit" ], "Orbit" );
        
        }, { size: ["20%", null], float: "left", draggable: false });

    },

    loadEnvironment( data ) {

        if( data.constructor === File )
        {
            const reader = new FileReader();
            reader.readAsBinaryString( data );
            reader.onload = e => this.loadEnvironmentBinary( e.target.result );
            return;
        }
        
        this.loadEnvironmentBinary( data );
    },

    loadEnvironmentBinary( bin ) {

        console.log( "Loading hdre binary", [ bin ] );

        // TODO
        // ...
    },

    loadGltf( data ) {

        if( data.constructor === File )
        {
            const reader = new FileReader();
            reader.readAsBinaryString( data );
            reader.onload = e => this.loadGltfBinary( e.target.result );
            return;
        }
        
        this.loadGltfBinary( data );
    },

    loadGltfBinary( bin ) {

        console.log( "Loading gltf binary", [ bin ] );

        // TODO
        // ...
    }

};

Rooms.initUI();