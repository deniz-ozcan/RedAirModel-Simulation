import pygame as pg
import numpy as np
import moderngl as mgl
import os
from jsbsim_gym.visualization.quaternion import Quaternion

dir_name = os.path.abspath(os.path.dirname(__file__))


def load_shader(ctx: mgl.Context, vertex_filename, frag_filename):
    """
    Bu Python fonksiyonu, bir OpenGL shader programını yüklemek için kullanılır. 
    Shader programı, genellikle bir vertex shader ve bir fragment shader'ı içerir. 
    Fonksiyon, verilen dosya adlarındaki vertex ve fragment shader kaynak kodlarını okur, OpenGL bağlamı (context) üzerinde bir shader programı oluşturur ve bu programı döndürür.

    Fonksiyonun parametreleri şunlardır:

    ctx: ModernGL kütüphanesi için bir bağlam nesnesi (context).
    vertex_filename: Vertex shader'ın kaynak kodunu içeren dosyanın adı.
    frag_filename: Fragment shader'ın kaynak kodunu içeren dosyanın adı.
    Fonksiyon, dosyaları okuyarak shader kaynak kodlarını elde eder ve bu kodları kullanarak bir shader programı oluşturur. 
    Oluşturulan shader programı, verilen ctx bağlamına aittir ve vertex ve fragment shader'ları içerir.

    Bu tür bir fonksiyon, genellikle grafik uygulamalarında, özellikle 3D rendering için OpenGL kullanıldığında shader programlarını yüklemek ve kullanmak için kullanılır. 
    Shader programları, GPU üzerinde çalışan özel programlar olup, görüntüleme işlemleri ve grafik efektlerinin uygulanmasında önemli bir rol oynar.
    """

    with open(os.path.join(dir_name, vertex_filename)) as f:
        vertex_src = f.read()
    with open(os.path.join(dir_name, frag_filename)) as f:
        frag_src = f.read()
    return ctx.program(vertex_shader=vertex_src, fragment_shader=frag_src)


def load_mesh(ctx: mgl.Context, program, filename):
    """
    Bu Python fonksiyonu, bir 3D modelin Wavefront OBJ dosyasından yüklenmesini sağlar. 
    Fonksiyon, modelin vertex (nokta), normal (yüzey normali) ve yüz (face) verilerini içeren bir OBJ dosyasını okur ve bu verileri kullanarak bir vertex array object (VAO) oluşturur.

    Fonksiyonun parametreleri şunlardır:

    ctx: ModernGL kütüphanesi için bir bağlam nesnesi (context).
    program: Modelin render edilmesi için kullanılacak shader program.
    filename: Yüklenmek istenen OBJ dosyasının adı.
    Fonksiyon, dosyayı okurken farklı satır türlerine göre işlem yapar:

    v: Vertex (nokta) bilgisi ekler.
    vn: Yüzey normali bilgisi ekler.
    f: Yüz (face) bilgisi ekler.
    Dosyanın diğer satır türleri (örneğin, vt, usemtl, mtllib) işleme dahil edilmez.

    Son olarak, oluşturulan vertex ve index verileri kullanılarak bir VAO oluşturulur ve bu VAO, verilen shader programına bağlanarak döndürülür.

    Bu tür bir fonksiyon genellikle 3D grafik uygulamalarında kullanılan model yükleme işlemleri için kullanılır. 
    Bu özel fonksiyon, özellikle ModernGL kütüphanesi kullanılarak OpenGL üzerinde çalışan bir uygulamada kullanılmak üzere tasarlanmış gibi görünüyor.
    """
    v = []
    vn = []
    vertices = []
    indices = []

    with open(os.path.join(dir_name, filename), 'r') as file:
        for line in file:
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                vertex = [float(val) for val in values[1:4]]
                v.append(vertex)
            elif values[0] == 'vn':
                norm = [float(val) for val in values[1:4]]
                vn.append(norm)
            elif values[0] == 'vt':
                continue
            elif values[0] in ('usemtl', 'usemat'):
                continue
            elif values[0] == 'mtllib':
                continue
            elif values[0] == 'f':
                for val in values[1:]:
                    w = val.split('/')
                    vertex = np.hstack((v[int(w[0]) - 1], vn[int(w[2]) - 1]))
                    vertices.append(vertex)
                start = len(vertices) - len(values) + 1
                for i in range(start, len(vertices) - 2):
                    indices.append([start, i + 1, i + 2])

    vbo = ctx.buffer(np.hstack(vertices).astype(np.float32).tobytes())
    ebo = ctx.buffer(np.hstack(indices).flatten().astype(np.uint32).tobytes())
    return ctx.simple_vertex_array(program, vbo, 'aPos', 'aNormal', index_buffer=ebo)


def perspective(fov, aspect, near, far):
    """
    Bu Python fonksiyonu, bir perspektif projeksiyon matrisini oluşturmak için kullanılır. Perspektif projeksiyon matrisi, 3D sahnedeki nesnelerin bir kameranın bakış açısından nasıl göründüğünü simgeler.
    Fonksiyonun parametreleri şunlardır:

    fov: Görüş açısı (Field of View) derece cinsinden. Fonksiyon, bu açıyı kullanarak içsel olarak radyan cinsine çevirir.
    aspect: Görüntü genişliği ile yüksekliği arasındaki oran.
    near: Kamera tarafından görülebilen en yakın nesnenin uzaklığı.
    far: Kamera tarafından görülebilen en uzak nesnenin uzaklığı.
    Fonksiyon, verilen parametrelerle perspektif projeksiyon matrisini oluşturur ve bu matrisi Numpy kütüphanesiyle temsil edilen bir dizi olarak döndürür.
    """
    fov *= np.pi / 180
    right = -np.tan(fov / 2) * near
    top = -right / aspect
    return np.array([[near / right, 0, 0, 0],
                     [0, near / top, 0, 0],
                     [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)],
                     [0, 0, 1, 0]], dtype=np.float32)


class Transform:
    """
    bir 3D dönüşümü temsil etmek için Transform adlı bir sınıf içerir. Bu sınıf, bir nesnenin pozisyonunu, dönüşünü (rotasyonunu) ve ölçeğini yönetir. 
    Bu sınıf, 3D sahnede nesnelerin pozisyonunu, dönüşünü ve ölçeğini yönetmek için kullanılır.
    Ayrıca, dönüşüm matrisi ve ters dönüşüm matrisi gibi önemli özellikleri sağlar. 
    Transform sınıfı, genellikle grafik programlamada nesnelerin dünyadaki konumlarını ve dönüşlerini temsil etmek için kullanılır.
    İşte sınıfın temel özellikleri:

    __init__ fonksiyonu, başlangıç durumunu ayarlar. Pozisyon varsayılan olarak (0, 0, 0) olarak ayarlanır, rotasyon varsayılan olarak bir kimlik dönüşü (Quaternion()) ve ölçek (scale) varsayılan olarak 1 olarak ayarlanır.
    Özellikler (Properties):

    position, x, y, z: Pozisyonu temsil eden özellikler. Pozisyonun her bir koordinatı ayrı ayrı get ve set edilebilir.
    rotation: Dönüşü (rotasyonu) temsil eden özellik. Dönüş, bir Quaternion nesnesi olarak saklanır.
    matrix: 3D dönüşüm matrisini döndüren özellik. Pozisyon, dönüş ve ölçek bilgilerini içerir.
    inv_matrix: Ters dönüşüm matrisini döndüren özellik. Nesnenin ters dönüşünü, ters pozisyonunu ve ters ölçeğini içerir.
    Metotlar:

    copy: Dönüşümün bir kopyasını oluşturur.
    """

    def __init__(self):
        self._position = np.zeros(3)
        self._rotation = Quaternion()
        self.scale = 1

    @property
    def position(self):
        return self._position.copy()

    @position.setter
    def position(self, position):
        self._position[:] = position

    @property
    def x(self):
        return self._position[0]

    @x.setter
    def x(self, x):
        self._position[0] = x

    @property
    def y(self):
        return self._position[1]

    @y.setter
    def y(self, y):
        self._position[1] = y

    @property
    def z(self):
        return self._position[2]

    @z.setter
    def z(self, z):
        self._position[2] = z

    @property
    def rotation(self):
        return self._rotation.copy()

    @rotation.setter
    def rotation(self, rotation):
        self._rotation._arr[:] = rotation._arr

    @property
    def matrix(self):
        matrix = np.eye(4)
        matrix[:3, :3] = self._rotation.mat().dot(np.eye(3) * self.scale)
        matrix[:3, 3] = self._position
        return matrix

    @property
    def inv_matrix(self):
        matrix = np.eye(4)
        matrix[:3, 3] = -self._position
        scale = np.eye(4)
        scale[:3, :3] /= self.scale
        matrix = scale.dot(matrix)
        rot = np.eye(4)
        rot[:3, :3] = self.rotation.inv().mat()
        matrix = rot.dot(matrix)
        return matrix


class RenderObject:
    """
    Bu sınıf, 3D sahnede render edilecek nesnelerin temel özelliklerini ve render işlemlerini sağlar. İşte sınıfın temel özellikleri:
    RenderObject Sınıfı (RenderObject):
    __init__ fonksiyonu, bir vao (vertex array object) parametresini alır ve sınıfın başlangıç durumunu ayarlar.
    color değişkeni, nesnenin rengini belirtir. Varsayılan olarak (1.0, 1.0, 1.0) yani beyaz renktir.
    transform değişkeni, nesnenin pozisyonunu ve dönüşünü temsil eden bir Transform nesnesini içerir.
    draw_mode değişkeni, nesnenin nasıl çizileceğini belirtir. Varsayılan olarak mgl.TRIANGLES kullanılır.
    render Fonksiyonu:
    Bu fonksiyon, nesneyi render etmek için kullanılır.
    self.vao.program['model'] ifadesi, nesnenin model matrisini shader programına gönderir. Bu, nesnenin pozisyonu ve dönüşünü içerir.
    self.vao.program['color'] ifadesi, nesnenin rengini shader programına gönderir.
    self.vao.render(self.draw_mode) ifadesi, vertex array object'in render işlemini başlatır. 
    self.draw_mode değişkeni, nesnenin çizim modunu belirtir (örneğin, üçgenler şeklinde).
    Bu sınıf, 3D sahnede çeşitli nesneleri temsil etmek ve bu nesneleri render etmek için temel özellikleri sağlar. 
    RenderObject sınıfından türetilen diğer sınıflar, özel davranışlar ekleyebilir veya bu sınıfın temel özelliklerini kullanabilir.
    """
    def __init__(self, vao):
        self.vao = vao
        self.color = 1.0, 1.0, 1.0
        self.transform = Transform()
        self.draw_mode = mgl.TRIANGLES

    def render(self):
        self.vao.program['model'] = tuple(np.hstack(self.transform.matrix.T))
        self.vao.program['color'] = self.color
        self.vao.render(self.draw_mode)


class Grid(RenderObject):
    """
    Bu Python kodu, bir Grid adlı sınıfı içerir. Bu sınıf, bir 3D ızgara oluşturur ve bu ızgarayı render etmek için ModernGL kütüphanesini kullanır. İşte sınıfın temel özellikleri:

    __init__ fonksiyonu, ızgara oluşturmak için gerekli hesaplamaları gerçekleştirir.
    low ve high değişkenleri, ızgaranın boyutlarına göre sınırları belirler.
    vertices ve indices listeleri, ızgara noktalarının ve çizgilerin tanımlanması için kullanılır.
    vertices listesi, ızgara noktalarını içerir.
    indices listesi, ızgara çizgilerini tanımlar.
    Noktaların ve çizgilerin tanımlandığı listeler birleştirilip düzenlenir.
    ModernGL buffer nesneleri oluşturularak bu veriler GPU'ya taşınır.
    vao (vertex array object), vbo (vertex buffer object) ve ebo (element buffer object) oluşturulur.
    draw_mode değişkeni, ızgaranın nasıl çizileceğini belirtir (bu durumda mgl.LINES ile çizilir).
    RenderObject Sınıfından Miras Alma:

    RenderObject sınıfından miras alınmıştır. RenderObject sınıfı, sahnede render edilecek nesnelerin temel özelliklerini içerir.
    Not:

    RenderObject sınıfının içeriği kod örneğinde verilmemiştir ancak bu sınıfın, sahnede render edilen nesnelerin temel özelliklerini (render fonksiyonu, pozisyon, dönme, vb.) içermesi beklenir.
    Bu Grid sınıfı, sahnede bir 3D ızgara oluşturmak ve bu ızgarayı render etmek için kullanılır. 
    Izgara, belirli bir boyutta ve belirli bir aralıkta (spacing) çizgiler içerir.
    Oluşturulan ızgara, daha büyük bir grafik sahnesinde kullanılabilir.
    """

    def __init__(self, ctx: mgl.Context, program, n, spacing):
        super().__init__(None)
        low = -(n - 1) * spacing / 2
        high = -low
        vertices = []
        indices = []
        for i in range(n):
            vertices.append([low + spacing * i, 0, low])
            vertices.append([low + spacing * i, 0, high])
            indices.append([i * 2, i * 2 + 1])
        for i in range(n):
            vertices.append([low, 0, low + spacing * i])
            vertices.append([high, 0, low + spacing * i])
            indices.append([n * 2 + i * 2, n * 2 + i * 2 + 1])
        vertices = np.hstack(vertices)
        indices = np.hstack(indices)
        vbo = ctx.buffer(vertices.astype(np.float32).tobytes())
        ebo = ctx.buffer(indices.astype(np.uint32).tobytes())
        self.vao = ctx.simple_vertex_array(program, vbo, 'aPos', index_buffer=ebo)
        self.draw_mode = mgl.LINES


class Viewer:
    """
    3D görüntüleyici (viewer) sınıfını ve bu sınıfın kullanımını içerir. İşte temel özellikleri:

    Viewer Sınıfı (Viewer):

    __init__ fonksiyonu, görüntüleyiciyi başlatır ve gerekli ayarları yapar.
    run fonksiyonu, görüntüleyiciyi çalıştırmak için bir döngü içinde çağrılır.
    callback fonksiyonu, kullanıcı tarafından tanımlanabilen bir geri çağrıdır (callback). Varsayılan olarak boştur.
    set_view fonksiyonu, görüntüleyicinin görüntüleme parametrelerini (pozisyon, rotasyon) ayarlar.
    get_frame fonksiyonu, görüntülenen kareyi alır.
    render fonksiyonu, sahnede bulunan nesneleri render eder ve ekrana görüntüyü çizer.
    close fonksiyonu, görüntüleyiciyi kapatır ve kaynakları serbest bırakır.
    Görüntüleme Konteksi ve Shader Ayarları:

    Görüntüleme konteksi oluşturulur (Pygame ve ModernGL kullanılarak).
    Shader programları (simple.vert, simple.frag, unlit.frag) yüklenir ve ayarları yapılır.
    Derinlik testi etkinleştirilir.
    Görüntüleme Döngüsü:

    run fonksiyonu içinde bir döngü oluşturulur.
    Döngü içinde, kullanıcı etkileşimleri kontrol edilir (QUIT event'ı dinlenir).
    Her döngüde callback fonksiyonu çağrılır.
    Sahnedeki nesneler render edilir (render fonksiyonu).
    Ekran güncellenir ve FPS (frame per second) kontrolü yapılır.
    Nesnelerin dönme hareketi için bir Quaternion kullanılır.
    Transform ve Kamera Ayarları:

    Transform sınıfı, nesnelerin pozisyon ve rotasyonunu temsil eder.
    Kamera ayarları ve görüntüleme matrisi (projection) oluşturulur.
    Sahnedeki Nesneler:

    objects listesi, sahnede render edilecek nesneleri içerir.
    Her bir nesne, sahnede render edilirken çağrılacak render fonksiyonuna sahip olmalıdır.
    """

    def __init__(self, width, height, fps=30, headless=False):
        self.transform = Transform()
        self.width = width
        self.height = height
        self.fps = fps

        if headless:
            self.ctx = mgl.create_standalone_context()
            self.display = None
            self.clock = None
        else:
            pg.init()
            pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
            pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
            pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)  # Core profilini kullanmak istiyorsanız
            pg.display.gl_set_attribute(pg.GL_MULTISAMPLEBUFFERS, 1)
            pg.display.gl_set_attribute(pg.GL_MULTISAMPLESAMPLES, 4)
            self.display = pg.display.set_mode((width, height), pg.DOUBLEBUF | pg.OPENGL)
            self.ctx = mgl.create_context()
            self.clock = pg.time.Clock()

        self.ctx.enable(mgl.DEPTH_TEST)

        self.projection = perspective(90, width / height, .1, 1000.)

        self.prog = load_shader(self.ctx, "simple.vert", "simple.frag")
        self.prog['projection'] = tuple(np.hstack(self.projection.T))
        self.prog['lightDir'] = .6, -.8, 1.0

        self.unlit = load_shader(self.ctx, "simple.vert", "unlit.frag")
        self.unlit['projection'] = tuple(np.hstack(self.projection.T))
        self.set_view()

        self.objects = []

    def run(self):
        running = True
        t = 0
        while running:
            pg.event.pump()
            camera_speed = .3
            camera_rot = np.pi / 36
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

            self.callback()

            self.render()
            self.clock.tick(self.fps)
            t += 1

            self.scene.root.children[0].transform.rotation = Quaternion.from_euler(0, t * np.pi / self.fps, 0)

        self.close()

    def callback(self):
        pass

    def set_view(self, x=None, y=None, z=None, rotation=None):
        if x is not None:
            self.transform.x = x
        if y is not None:
            self.transform.y = y
        if z is not None:
            self.transform.z = z
        if rotation is not None:
            self.transform.rotation = rotation
        self.prog['view'] = tuple(np.hstack(self.transform.inv_matrix.T))
        self.unlit['view'] = tuple(np.hstack(self.transform.inv_matrix.T))

    def get_frame(self):
        data = self.ctx.fbo.read()
        return np.array(bytearray(data)).reshape(self.height, self.width, 3)[-1::-1, :, :]

    def render(self):
        pg.event.pump()
        self.ctx.clear(0.5, 0.5, 0.5, 1.0)

        for obj in self.objects:
            obj.render()

        pg.display.flip()

    def close(self):
        pg.quit()


if __name__ == "__main__":
    """
    Bu kod bloğu, Transform sınıfının bir örneğini oluşturur, bu örneği belirli bir pozisyon, rotasyon ve ölçekle ayarlar, ardından bu transformasyonun tersini (inv_matrix) ve doğrudan tersini (inv(trans.matrix)) hesaplar. 
    Sonrasında, bu iki matrisin öğelerinin mutlak değerlerinin toplamını kontrol eder ve bu değerin sıfıra yakın olup olmadığını kontrol eder.

    Bu tür bir kontrol, bir matrisin tersini doğru bir şekilde hesaplayıp hesaplamadığını kontrol etmek için kullanılabilir. 
    np.sum(np.abs(trans.inv_matrix) - np.abs(inv(trans.matrix))) ifadesi, iki matrisin öğelerinin mutlak değerlerini alıp farklarını hesaplar ve bu farkların toplamını döndürür. Eğer bu toplam sıfıra yakınsa, matrislerin tersleri birbirine eşittir.

    Not: Matris hesaplamalarında kayan nokta hassasiyeti nedeniyle, tam olarak sıfıra eşit olmayan değerlerin ortaya çıkması mümkündür. 
    Bu nedenle, sıfıra çok yakın bir değerle karşılaştırma (np.allclose gibi) genellikle daha güvenilirdir.
    """

    from numpy.linalg import inv

    trans = Transform()
    trans.position = -2, 2, 3
    trans.rotation = Quaternion.from_euler(-.5, -.2, .3)
    trans.scale = 5.0

    print(np.sum(np.abs(trans.inv_matrix) - np.abs(inv(trans.matrix))))