
function _processVector( vector )
{
    var array = [];
    for( var i = 0; i < vector.size(); ++i )
        array.push( vector.get(i) );

    return array;
}

window.App = {

    dragSupportedExtensions: [ /*'glb', 'ply',*/ 'room' ],


    init() {

        this.initUI();
    },

    initUI() {

        var canvas = document.getElementById( "canvas" );
        document.body.appendChild(canvas);

        document.body.addEventListener('dragenter', e => e.preventDefault() );
        document.body.addEventListener('dragleave', e => e.preventDefault());
        document.body.addEventListener("dragover", function (event) {
            // prevent default to allow drop
            event.preventDefault();
        }, false);
        document.body.addEventListener('drop', (e) => {
            e.preventDefault();
            //this.toggleModal( true );
            const file = e.dataTransfer.files[0];
            const ext = this.getExtension( file.name );
            switch(ext)
            {
                // case "glb": this.loadFile(this._loadGltf, file); break;
                // case "ply": this.loadFile(this._loadPly, file); break;
                case "room": this.loadFile(this._loadRoom, file); break;
            }
        });

        // Create loading  modal

        this.modal = document.createElement( 'div' );

        this.modal.style.width = "100%";
        this.modal.style.height = "100%";
        this.modal.style.opacity = "0.9";
        this.modal.style.backgroundColor = "#000";
        this.modal.style.position = "absolute";
        this.modal.style.cursor = "wait";
        this.modal.hidden = true;

        document.body.appendChild( this.modal );
    },

    getExtension( filename ) {

        return filename.includes('.') ? filename.split('.').pop() : null;
    },

    toggleModal( force ) {

        this.modal.hidden = force !== undefined ? (!force) : !this.modal.hidden;
    },

    loadFile( loader, file, data ) {

        if( !data )
        {
            // file is the path URL
            if( file.constructor == String )
            {
                const path = file;
                LX.requestBinary( path, ( data ) => loader.call(this, path, data ), ( e ) => {
                    LX.popup( e.constructor === String ? e :  `[${ path }] can't be loaded.`, "Request Blocked", { size: ["400px", "auto"], timeout: 10000 } );
                    this.toggleModal( false );
                } );
                return;
            }

            const reader = new FileReader();
            reader.readAsArrayBuffer( file );
            reader.onload = e => loader.call(this, file.name, e.target.result, file);
            reader.onerror = (e, a, b, c) => {
                debugger;
            };

            return;
        }
        
        loader.call(this, file.name ?? file, data );
    },

   
    // _loadGltf( name, buffer ) {

    //     name = name.substring( name.lastIndexOf( '/' ) + 1 );
        
    //     console.log( "Loading glb", [ name, buffer ] );

    //     this._fileStore( name, buffer );

    //     window.engineInstance.appendGLB( name );

    //     this.toggleModal( false );
    // },

    // _loadPly( name, buffer ) {

    //     name = name.substring( name.lastIndexOf( '/' ) + 1 );

    //     console.log( "Loading ply", [ name, buffer ] );

    //     this._fileStore( name, buffer );

    //     window.engineInstance.loadPly( name );

    //     this.toggleModal( false );
    // },

    _loadRoom( name, buffer, file ) {

        name = name.substring( name.lastIndexOf( '/' ) + 1 );

        console.log( "Loading ROOM", [ name, buffer ] );

        this._fileStore( name, buffer, file );

        window.engineInstance.loadRoom( name );

        this.toggleModal( false );
    },

    _fileStore( filename, buffer, file ) {

        // debugger;
        // let data = new Uint8Array( buffer );
        // let stream = FS.open( filename, 'w+' );
        FS.writeFile( filename, new Int8Array( buffer ), {
            flags: 'w',
            encoding: 'binary',
            canOwn: true
        } );
        // FS.write( stream, data, 0, data.length, 0 );
        // FS.close( stream );

        console.log('File exists:', FS.analyzePath('/' + filename).exists);
        console.log('File size:', FS.stat('/' + filename).size);
    }
};

window.App.init();